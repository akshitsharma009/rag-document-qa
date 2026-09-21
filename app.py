import hashlib
import json
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

from src.vector_store import (
    save_vectorstore,
    load_vectorstore,
    vectorstore_exists,
    get_vectorstore_path
)


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Document Question Answering")

st.caption(
    "Ask questions about your PDF documents using "
    "retrieval-augmented generation."
)


# ============================================================
# CONSTANTS
# ============================================================

DOCUMENTS_BASE_PATH = Path(
    "data/documents"
)

TOP_K = 3

# Minimum relevance score required before
# sending retrieved context to the LLM.
RELEVANCE_THRESHOLD = 0.35


# ============================================================
# FALLBACK RESPONSE
# ============================================================

FALLBACK_MESSAGE = (
    "I couldn't find sufficient information in the "
    "uploaded documents to answer this question."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading embedding model..."
)
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )


# ============================================================
# LOAD FLAN-T5 MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading FLAN-T5 model..."
)
def load_llm():

    text_generation_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        do_sample=False
    )

    return HuggingFacePipeline(
        pipeline=text_generation_pipeline
    )


# ============================================================
# RAG PROMPT
# ============================================================

RAG_PROMPT = """
You are a document question-answering assistant.

Answer the question using ONLY the information
provided in the context below.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not make assumptions.
4. Answer only from the provided context.
5. Keep the answer clear and concise.

If the context does not contain enough information
to answer the question, respond exactly with:

I couldn't find sufficient information in the uploaded documents to answer this question.

Context:
{context}

Question:
{question}

Answer:
"""


# ============================================================
# CREATE DOCUMENT COLLECTION ID
# ============================================================

def create_collection_id(
    uploaded_files
):
    """
    Create a deterministic ID for the exact
    set of uploaded documents.

    The same documents produce the same ID.
    """

    hasher = hashlib.sha256()

    files_data = sorted(
        [
            (
                uploaded_file.name,
                uploaded_file.getvalue()
            )
            for uploaded_file in uploaded_files
        ],
        key=lambda item: item[0]
    )

    for filename, file_bytes in files_data:

        hasher.update(
            filename.encode("utf-8")
        )

        hasher.update(
            file_bytes
        )

    return hasher.hexdigest()[:16]


# ============================================================
# SAVE UPLOADED DOCUMENTS
# ============================================================

def save_uploaded_documents(
    uploaded_files,
    collection_id
):
    """
    Save uploaded PDFs under their
    document collection directory.
    """

    collection_path = (
        DOCUMENTS_BASE_PATH
        / collection_id
    )

    collection_path.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_paths = []

    for uploaded_file in uploaded_files:

        file_path = (
            collection_path
            / uploaded_file.name
        )

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getvalue()
            )

        saved_paths.append(
            file_path
        )

    return saved_paths


# ============================================================
# SAVE COLLECTION METADATA
# ============================================================

def save_collection_metadata(
    collection_id,
    uploaded_files,
    page_count,
    chunk_count
):
    """
    Save metadata describing the document collection.
    """

    vectorstore_path = get_vectorstore_path(
        collection_id
    )

    vectorstore_path.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {
        "collection_id": collection_id,
        "files": [
            uploaded_file.name
            for uploaded_file in uploaded_files
        ],
        "page_count": page_count,
        "chunk_count": chunk_count,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "embedding_model": (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        "generation_model": (
            "google/flan-t5-base"
        ),
        "top_k": TOP_K,
        "relevance_threshold": (
            RELEVANCE_THRESHOLD
        )
    }

    metadata_path = (
        vectorstore_path
        / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


# ============================================================
# LOAD COLLECTION METADATA
# ============================================================

def load_collection_metadata(
    collection_id
):
    """
    Load metadata for an existing collection.
    """

    metadata_path = (
        get_vectorstore_path(
            collection_id
        )
        / "metadata.json"
    )

    if not metadata_path.exists():

        return None

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vectorstore(
    uploaded_files,
    embeddings,
    collection_id
):
    """
    Process uploaded PDFs and create a
    persistent FAISS vector store.
    """

    documents = []

    # --------------------------------------------------------
    # Save original PDFs
    # --------------------------------------------------------

    saved_paths = save_uploaded_documents(
        uploaded_files,
        collection_id
    )

    # --------------------------------------------------------
    # Load PDFs
    # --------------------------------------------------------

    for file_path, uploaded_file in zip(
        saved_paths,
        uploaded_files
    ):

        loader = PyPDFLoader(
            str(file_path)
        )

        docs = loader.load()

        for doc in docs:

            doc.metadata[
                "source_file"
            ] = uploaded_file.name

        documents.extend(
            docs
        )

    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
    )

    split_docs = (
        text_splitter.split_documents(
            documents
        )
    )

    # --------------------------------------------------------
    # Create FAISS
    # --------------------------------------------------------

    vectorstore = (
        FAISS.from_documents(
            split_docs,
            embeddings
        )
    )

    # --------------------------------------------------------
    # Save FAISS
    # --------------------------------------------------------

    save_vectorstore(
        vectorstore,
        collection_id
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_collection_metadata(
        collection_id=collection_id,
        uploaded_files=uploaded_files,
        page_count=len(documents),
        chunk_count=len(split_docs)
    )

    return (
        vectorstore,
        documents,
        split_docs
    )


# ============================================================
# LOAD OR CREATE VECTOR STORE
# ============================================================

def get_vectorstore(
    uploaded_files,
    embeddings
):
    """
    Load an existing vector store if the exact
    document collection has already been processed.

    Otherwise create a new vector store.
    """

    collection_id = (
        create_collection_id(
            uploaded_files
        )
    )

    # --------------------------------------------------------
    # Existing collection
    # --------------------------------------------------------

    if vectorstore_exists(
        collection_id
    ):

        vectorstore = (
            load_vectorstore(
                embeddings,
                collection_id
            )
        )

        metadata = (
            load_collection_metadata(
                collection_id
            )
        )

        return (
            vectorstore,
            collection_id,
            metadata,
            True
        )

    # --------------------------------------------------------
    # New collection
    # --------------------------------------------------------

    (
        vectorstore,
        documents,
        split_docs
    ) = create_vectorstore(
        uploaded_files,
        embeddings,
        collection_id
    )

    metadata = (
        load_collection_metadata(
            collection_id
        )
    )

    return (
        vectorstore,
        collection_id,
        metadata,
        False
    )


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_relevant_documents(
    vectorstore,
    query
):
    """
    Retrieve the top-k documents together with
    their relevance scores.

    Only documents above the configured threshold
    are considered relevant.
    """

    scored_documents = (
        vectorstore
        .similarity_search_with_relevance_scores(
            query,
            k=TOP_K
        )
    )

    relevant_documents = []

    for document, score in scored_documents:

        if score >= RELEVANCE_THRESHOLD:

            relevant_documents.append(
                (
                    document,
                    score
                )
            )

    return relevant_documents


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(relevant_documents):
    context_parts = []

    for doc, score in relevant_documents:
        context_parts.append(doc.page_content)

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    llm,
    query,
    relevant_documents
):
    """
    Generate an answer using only the
    retrieved relevant document context.
    """

    if not relevant_documents:

        return FALLBACK_MESSAGE

    context = build_context(
        relevant_documents
    )

    prompt = RAG_PROMPT.format(
        context=context,
        question=query
    )

    answer = llm.invoke(
        prompt
    )

    if answer is None:

        return FALLBACK_MESSAGE

    answer = str(
        answer
    ).strip()

    if not answer:

        return FALLBACK_MESSAGE

    return answer


# ============================================================
# PDF UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_files:

    try:

        # ====================================================
        # LOAD EMBEDDINGS
        # ====================================================

        embeddings = load_embeddings()


        # ====================================================
        # LOAD OR CREATE VECTOR STORE
        # ====================================================

        with st.spinner(
            "Checking document knowledge base..."
        ):

            (
                vectorstore,
                collection_id,
                metadata,
                was_loaded
            ) = get_vectorstore(
                uploaded_files,
                embeddings
            )


        # ====================================================
        # LOAD LANGUAGE MODEL
        # ====================================================

        llm = load_llm()


        # ====================================================
        # STORE IN SESSION
        # ====================================================

        st.session_state[
            "vectorstore"
        ] = vectorstore

        st.session_state[
            "llm"
        ] = llm

        st.session_state[
            "collection_id"
        ] = collection_id

        st.session_state[
            "metadata"
        ] = metadata


        # ====================================================
        # KNOWLEDGE BASE STATUS
        # ====================================================

        if was_loaded:

            st.success(
                "Existing FAISS knowledge base "
                "loaded successfully."
            )

        else:

            st.success(
                "New FAISS knowledge base created "
                "and saved successfully."
            )


        # ====================================================
        # DOCUMENT PROCESSING DETAILS
        # ====================================================

        with st.expander(
            "📊 Document Processing Details"
        ):

            st.write(
                f"**Documents:** "
                f"{len(uploaded_files)}"
            )

            st.write(
                f"**Pages processed:** "
                f"{metadata.get('page_count', 'N/A')}"
            )

            st.write(
                f"**Text chunks:** "
                f"{metadata.get('chunk_count', 'N/A')}"
            )

            st.write(
                "**Chunk size:** 500 characters"
            )

            st.write(
                "**Chunk overlap:** 100 characters"
            )

            st.write(
                "**Embedding model:** "
                "sentence-transformers/all-MiniLM-L6-v2"
            )

            st.write(
                "**Vector database:** FAISS"
            )

            st.write(
                f"**Retrieved chunks:** Top {TOP_K}"
            )

            st.write(
                f"**Minimum relevance score:** "
                f"{RELEVANCE_THRESHOLD}"
            )

            st.write(
                "**Generation model:** "
                "google/flan-t5-base"
            )

            st.write(
                "**Device:** CPU"
            )

            st.write(
                "**Storage:** Persistent"
            )

            st.write(
                f"**Collection ID:** "
                f"`{collection_id}`"
            )


        # ====================================================
        # DOCUMENTS IN KNOWLEDGE BASE
        # ====================================================

        st.subheader(
            "📚 Documents in Knowledge Base"
        )

        for uploaded_file in uploaded_files:

            st.write(
                f"📄 {uploaded_file.name}"
            )


        # ====================================================
        # ASK QUESTION
        # ====================================================

        st.divider()

        st.subheader(
            "💬 Ask a Question"
        )

        query = st.text_input(
            "Ask something about your uploaded documents:",
            placeholder=(
                "Example: What is the main objective "
                "of this paper?"
            )
        )


        # ====================================================
        # QUESTION ANSWERING
        # ====================================================

        if query:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                try:

                    # -----------------------------------------
                    # RETRIEVE TOP-K DOCUMENTS + SCORES
                    # -----------------------------------------

                    relevant_documents = (
                        retrieve_relevant_documents(
                            st.session_state[
                                "vectorstore"
                            ],
                            query
                        )
                    )


                    # -----------------------------------------
                    # SHOW RETRIEVAL STATUS
                    # -----------------------------------------

                    st.subheader(
                        "🔎 Retrieval Analysis"
                    )

                    if relevant_documents:

                        st.success(
                            f"Found "
                            f"{len(relevant_documents)} "
                            f"relevant chunk(s)."
                        )

                    else:

                        st.warning(
                            "No sufficiently relevant "
                            "document chunks were found."
                        )


                    # -----------------------------------------
                    # GENERATE ANSWER
                    # -----------------------------------------

                    answer = generate_answer(
                        st.session_state[
                            "llm"
                        ],
                        query,
                        relevant_documents
                    )


                    # -----------------------------------------
                    # DISPLAY ANSWER
                    # -----------------------------------------

                    st.subheader(
                        "🧠 Answer"
                    )

                    st.write(
                        answer
                    )


                    # -----------------------------------------
                    # DISPLAY RETRIEVAL SCORES
                    # -----------------------------------------

                    if relevant_documents:

                        with st.expander(
                            "📊 Retrieval Relevance Scores"
                        ):

                            for index, (
                                document,
                                score
                            ) in enumerate(
                                relevant_documents,
                                start=1
                            ):

                                st.write(
                                    f"**Chunk {index}: "
                                    f"{score:.4f}**"
                                )


                    # -----------------------------------------
                    # DISPLAY SOURCES
                    # -----------------------------------------

                    st.subheader(
                        "📚 Sources"
                    )

                    if relevant_documents:

                        for index, (
                            document,
                            score
                        ) in enumerate(
                            relevant_documents,
                            start=1
                        ):

                            source_file = (
                                document.metadata.get(
                                    "source_file",
                                    document.metadata.get(
                                        "source",
                                        "Unknown document"
                                    )
                                )
                            )

                            page_number = (
                                document.metadata.get(
                                    "page",
                                    None
                                )
                            )

                            if page_number is not None:

                                page_number += 1


                            if page_number is not None:

                                source_label = (
                                    f"Source {index}: "
                                    f"{source_file} "
                                    f"(Page {page_number}) "
                                    f"| Relevance: "
                                    f"{score:.4f}"
                                )

                            else:

                                source_label = (
                                    f"Source {index}: "
                                    f"{source_file} "
                                    f"| Relevance: "
                                    f"{score:.4f}"
                                )


                            with st.expander(
                                source_label
                            ):

                                st.write(
                                    document.page_content
                                )

                    else:

                        st.info(
                            "No source documents were "
                            "relevant enough to display."
                        )


                except Exception as e:

                    st.error(
                        "An error occurred while "
                        f"answering the question:\n\n{e}"
                    )


    except Exception as e:

        st.error(
            "An error occurred while preparing "
            f"the document knowledge base:\n\n{e}"
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👆 Upload one or more PDF documents "
        "to get started."
    )

    st.markdown(
        """
        ### How it works

        1. Upload one or more PDF documents.
        2. Generate a unique collection ID.
        3. Check whether a FAISS knowledge base already exists.
        4. Load the existing FAISS index when available.
        5. Otherwise extract and chunk the PDFs.
        6. Generate semantic embeddings.
        7. Create and persist a new FAISS index.
        8. Retrieve the top 3 chunks with relevance scores.
        9. Filter out low-relevance chunks.
        10. Pass only relevant context to FLAN-T5.
        11. Generate a grounded answer.
        12. Display the answer with source pages.
        """
    )