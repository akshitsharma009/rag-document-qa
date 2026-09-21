import sys
from pathlib import Path
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_huggingface import HuggingFaceEmbeddings

from src.vector_store import load_vectorstore


# -----------------------------
# Configuration
# -----------------------------

VECTORSTORE_BASE_PATH = PROJECT_ROOT / "data" / "vectorstore"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"

TOP_K = 3

# Lower L2 distance = more similar.
# This value will be validated through evaluation.
DISTANCE_THRESHOLD = 1.30


# -----------------------------
# Load evaluation questions
# -----------------------------

def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


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

    if len(collections) > 1:
        print("Multiple vector-store collections found.")
        print("Using:", collections[0].name)

    return collections[0].name


# -----------------------------
# Load embedding model
# -----------------------------

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# Retrieve using raw FAISS distance
# -----------------------------

def retrieve_documents(vectorstore, query):
    results = vectorstore.similarity_search_with_score(
        query,
        k=TOP_K
    )

    return results


# -----------------------------
# Evaluate a single question
# -----------------------------

def evaluate_question(vectorstore, question):
    results = retrieve_documents(
        vectorstore,
        question
    )

    relevant_results = [
        (doc, distance)
        for doc, distance in results
        if distance <= DISTANCE_THRESHOLD
    ]

    predicted_relevant = len(relevant_results) > 0

    return {
        "predicted_relevant": predicted_relevant,
        "results": results,
        "relevant_results": relevant_results
    }


# -----------------------------
# Calculate metrics
# -----------------------------

def calculate_metrics(results):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in results:
        expected = result["expected_relevant"]
        predicted = result["predicted_relevant"]

        if expected and predicted:
            true_positive += 1

        elif not expected and not predicted:
            true_negative += 1

        elif not expected and predicted:
            false_positive += 1

        elif expected and not predicted:
            false_negative += 1

    total = len(results)

    accuracy = (
        (true_positive + true_negative) / total
        if total
        else 0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0
    )

    fallback_accuracy = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive)
        else 0
    )

    return {
        "total": total,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fallback_accuracy": fallback_accuracy
    }


# -----------------------------
# Main evaluation
# -----------------------------

def main():
    print("=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"\nDistance threshold: {DISTANCE_THRESHOLD}")

    collection_id = find_collection_id()

    if collection_id is None:
        print("\nERROR: No vector store found.")
        print(
            "Upload and index a document through "
            "the Streamlit app first."
        )
        return

    print(f"Collection ID: {collection_id}")

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

    questions = load_questions()

    evaluation_results = []

    print("\nRunning evaluation...\n")

    for item in questions:
        question = item["question"]
        expected = item["expected_relevant"]

        evaluation = evaluate_question(
            vectorstore,
            question
        )

        predicted = evaluation["predicted_relevant"]

        top_distance = (
            evaluation["results"][0][1]
            if evaluation["results"]
            else None
        )

        relevant_count = len(
            evaluation["relevant_results"]
        )

        evaluation_results.append({
            "id": item["id"],
            "question": question,
            "expected_relevant": expected,
            "predicted_relevant": predicted,
            "top_distance": top_distance,
            "relevant_chunks": relevant_count
        })

        status = (
            "PASS"
            if expected == predicted
            else "FAIL"
        )

        print(f"[{status}] {question}")
        print(f"  Expected relevant : {expected}")
        print(f"  Predicted relevant: {predicted}")

        if top_distance is not None:
            print(
                f"  Top FAISS distance: "
                f"{top_distance:.4f}"
            )

        print(
            f"  Relevant chunks: "
            f"{relevant_count}"
        )

        print()

    metrics = calculate_metrics(
        evaluation_results
    )

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nTotal questions     : "
        f"{metrics['total']}"
    )

    print(
        f"True positives      : "
        f"{metrics['true_positive']}"
    )

    print(
        f"True negatives      : "
        f"{metrics['true_negative']}"
    )

    print(
        f"False positives     : "
        f"{metrics['false_positive']}"
    )

    print(
        f"False negatives     : "
        f"{metrics['false_negative']}"
    )

    print(
        f"\nAccuracy            : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision           : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall              : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Fallback accuracy  : "
        f"{metrics['fallback_accuracy']:.4f}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()