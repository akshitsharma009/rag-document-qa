import sys
from pathlib import Path
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

from src.vector_store import load_vectorstore


# -----------------------------
# Configuration
# -----------------------------

VECTORSTORE_BASE_PATH = PROJECT_ROOT / "data" / "vectorstore"

TOP_K = 2

# Lower FAISS distance = more similar
DISTANCE_THRESHOLD = 1.30

FALLBACK_MESSAGE = (
    "I couldn't find sufficient information in the uploaded "
    "documents to answer this question."
)


# -----------------------------
# Questions for answer evaluation
# -----------------------------

QUESTIONS = [
    {
        "id": 1,
        "question": "What is multi-head attention?",
        "expected_type": "answer"
    },
    {
        "id": 2,
        "question": "What is the architecture of the Transformer?",
        "expected_type": "answer"
    },
    {
        "id": 3,
        "question": "What is the role of the encoder in the Transformer?",
        "expected_type": "answer"
    },
    {
        "id": 4,
        "question": "What is the role of the decoder in the Transformer?",
        "expected_type": "answer"
    },
    {
        "id": 5,
        "question": "Why does the Transformer use self-attention?",
        "expected_type": "answer"
    },
    {
        "id": 6,
        "question": "What is the capital of France?",
        "expected_type": "fallback"
    },
    {
        "id": 7,
        "question": "Who invented Python?",
        "expected_type": "fallback"
    },
    {
        "id": 8,
        "question": "What is the population of India?",
        "expected_type": "fallback"
    }
]


# -----------------------------
# Find vector-store collection
# -----------------------------

def find_collection_id():
    if not VECTORSTORE_BASE_PATH.exists():
        return None

    collections = [
        path
        for path in VECTORSTORE_BASE_PATH.iterdir()
        if path.is_dir()
    ]

    if not collections:
        return None

    return collections[0].name


# -----------------------------
# Load embeddings
# -----------------------------

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# Load FLAN-T5
# -----------------------------

def load_llm():
    text2text_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        do_sample=False
    )

    return HuggingFacePipeline(
        pipeline=text2text_pipeline
    )


# -----------------------------
# Retrieve relevant documents
# -----------------------------

def retrieve_context(vectorstore, question):
    results = vectorstore.similarity_search_with_score(
        question,
        k=TOP_K
    )

    relevant_results = [
        (doc, distance)
        for doc, distance in results
        if distance <= DISTANCE_THRESHOLD
    ]

    return relevant_results


# -----------------------------
# Build context
# -----------------------------

def build_context(relevant_results):
    context_parts = []

    for doc, distance in relevant_results[:2]:
        context_parts.append(doc.page_content)

    return "\n\n".join(context_parts)


# -----------------------------
# Generate grounded answer
# -----------------------------

def generate_answer(llm, question, context):

    prompt = f"""
You are a document question-answering assistant.

Your task is to answer the user's question using ONLY the
information contained in the provided context.

STRICT RULES:
1. Answer the question directly.
2. Use only information supported by the context.
3. Do not use outside knowledge.
4. Do not invent or assume facts.
5. Do not copy the context word-for-word unless necessary.
6. Do not include source names, page numbers, metadata,
   citations, or labels such as "Source:" in the answer.
7. Do not mention the retrieval process.
8. Keep the answer concise, preferably 1 to 3 sentences.
9. If the context does not contain enough information,
   respond exactly with:

{FALLBACK_MESSAGE}

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(prompt)

    # Clean accidental whitespace
    answer = answer.strip()

    return answer

# -----------------------------
# Check fallback
# -----------------------------

def is_fallback(answer):
    normalized_answer = answer.lower().strip()

    return normalized_answer == FALLBACK_MESSAGE.lower()


# -----------------------------
# Main evaluation
# -----------------------------

def main():

    print("=" * 60)
    print("RAG ANSWER QUALITY EVALUATION")
    print("=" * 60)

    collection_id = find_collection_id()

    if collection_id is None:
        print("\nERROR: No vector store found.")
        return

    print(f"\nCollection ID: {collection_id}")

    print("\nLoading embedding model...")
    embeddings = load_embeddings()

    print("Loading FAISS vector store...")
    vectorstore = load_vectorstore(
        embeddings,
        collection_id
    )

    if vectorstore is None:
        print("\nERROR: Could not load vector store.")
        return

    print("\nLoading FLAN-T5...")
    llm = load_llm()

    print("\nRunning answer evaluation...\n")

    results = []

    for item in QUESTIONS:

        question = item["question"]
        expected_type = item["expected_type"]

        print("-" * 60)
        print(f"Question {item['id']}: {question}")

        relevant_results = retrieve_context(
            vectorstore,
            question
        )

        # -----------------------------
        # No relevant context
        # -----------------------------

        if not relevant_results:

            answer = FALLBACK_MESSAGE

            predicted_type = "fallback"

            print("\nRetrieved context: NONE")
            print(f"Answer: {answer}")

        # -----------------------------
        # Relevant context exists
        # -----------------------------

        else:

            context = build_context(
                relevant_results
            )

            answer = generate_answer(
                llm,
                question,
                context
            )

            predicted_type = (
                "fallback"
                if is_fallback(answer)
                else "answer"
            )

            print(
                f"\nRetrieved chunks: "
                f"{len(relevant_results)}"
            )

            print(
                f"Best FAISS distance: "
                f"{relevant_results[0][1]:.4f}"
            )

            print(f"\nAnswer: {answer}")

        passed = predicted_type == expected_type

        results.append({
            "id": item["id"],
            "question": question,
            "expected_type": expected_type,
            "predicted_type": predicted_type,
            "passed": passed,
            "answer": answer
        })

        print(
            f"\nResult: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # -----------------------------
    # Summary
    # -----------------------------

    total = len(results)

    passed = sum(
        result["passed"]
        for result in results
    )

    failed = total - passed

    fallback_tests = [
        result
        for result in results
        if result["expected_type"] == "fallback"
    ]

    fallback_passed = sum(
        result["passed"]
        for result in fallback_tests
    )

    fallback_accuracy = (
        fallback_passed / len(fallback_tests)
        if fallback_tests
        else 0
    )

    print("\n" + "=" * 60)
    print("ANSWER EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTotal questions    : {total}")
    print(f"Passed             : {passed}")
    print(f"Failed             : {failed}")

    print(
        f"Answer/Fallback accuracy: "
        f"{passed / total:.4f}"
    )

    print(
        f"Fallback accuracy       : "
        f"{fallback_accuracy:.4f}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()