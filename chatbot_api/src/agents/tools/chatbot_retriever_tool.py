from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document


def get_retriever(vector_store: Qdrant, collection_name: str) -> EnsembleRetriever:
    '''
    Creates an ensemble retriever combining vector-based (Qdrant) and keyword-based (BM25) retrieval methods.

    Args:
        vector_store: A Qdrant vector store instance.
        collection_name: The name of the collection in the Qdrant vector store.

    Returns:
        An EnsembleRetriever instance combining Qdrant and BM25 retrievers, or a BM25 retriever with a default error message if an error occurs.
    '''
    try:
        # Create vector retriever
        qdrant_retriever = vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={"k": 4},
        )

        # Create BM25 retriever from all documents
        documents=[
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vector_store.similarity_search("", k = 100)
        ]

        if not documents:
            raise ValueError(f"Don't find documents in collection '{collection_name}' ")

        bm25_retriever = BM25Retriever.from_documents(documents=documents)
        bm25_retriever.k=4

        # Ensemble all retrievers with weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[qdrant_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )

        return ensemble_retriever

    except Exception as e:
        print(f"Error during retriever initialization: {str(e)}")
        # Return a retriever with a default document if there is an error
        default_doc = [
            Document(
                page_content="An error occurred while connecting to the database. Please try again later.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)