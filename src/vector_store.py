from pathlib import Path

from langchain_community.vectorstores import FAISS


# ============================================================
# BASE VECTOR STORE DIRECTORY
# ============================================================

VECTORSTORE_BASE_PATH = Path("data/vectorstore")


# ============================================================
# GET VECTOR STORE PATH
# ============================================================

def get_vectorstore_path(collection_id):
    """
    Return the directory where a specific
    document collection's FAISS index is stored.
    """

    return VECTORSTORE_BASE_PATH / collection_id


# ============================================================
# CHECK IF VECTOR STORE EXISTS
# ============================================================

def vectorstore_exists(collection_id):
    """
    Check whether a persistent FAISS index
    exists for the given document collection.
    """

    vectorstore_path = get_vectorstore_path(
        collection_id
    )

    index_file = vectorstore_path / "index.faiss"
    metadata_file = vectorstore_path / "index.pkl"

    return (
        index_file.exists()
        and metadata_file.exists()
    )


# ============================================================
# SAVE VECTOR STORE
# ============================================================

def save_vectorstore(
    vectorstore,
    collection_id
):
    """
    Save a FAISS vector store for a specific
    document collection.
    """

    vectorstore_path = get_vectorstore_path(
        collection_id
    )

    vectorstore_path.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(vectorstore_path)
    )


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vectorstore(
    embeddings,
    collection_id
):
    """
    Load an existing FAISS vector store.

    Returns:
        FAISS vector store if it exists.
        None otherwise.
    """

    if not vectorstore_exists(
        collection_id
    ):
        return None

    vectorstore_path = get_vectorstore_path(
        collection_id
    )

    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore