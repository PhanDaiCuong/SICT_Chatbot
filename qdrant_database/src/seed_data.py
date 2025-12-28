import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient, models
from uuid import uuid4
import logging

LOGGER = logging.getLogger(__name__)


def load_data_from_local(filename:str, directory: str) -> tuple:
    '''
        Loads data from a local JSON file.

        Args:
            filename (str): The name of the JSON file to load (e.g., 'data.json').
            directory (str): The directory containing the file (e.g., 'data_v3').

        Returns:
            tuple: Returns a tuple containing (data, doc_name) where:
                - data: The parsed JSON data.
                - doc_name: The processed document name (the filename without the '.json' extension and with '_' replaced by spaces).
    '''
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data, filename.rsplit('.', 1)[0].replace('_', ' ')


def initial_qdrant_database(qdrant_host: str, qdrant_api_key: str, qdrant_collection_name: str) -> QdrantClient:
    '''
        Initializes a Qdrant client instance.

        Args:
            qdrant_host (str): The host address of the Qdrant database.
            qdrant_api_key (str): The API key for authenticating with the Qdrant database.

        Returns:
            QdrantClient: An initialized Qdrant client object.
    '''
    # Initial to instance client
    client = QdrantClient(
        url = qdrant_host,
        api_key=qdrant_api_key,
    )

    # setup dimension of embedding models and similarity algorithm
    vector_config = models.VectorParams(
        size=768,
        distance=models.Distance.COSINE
    )

    # Check existing collections 
    collections = client.get_collections().collections
    existing_collections = [collection.name for collection in collections]

    if qdrant_collection_name not in existing_collections:
        # If it don't create
        client.create_collection(
            collection_name=qdrant_collection_name,
            vectors_config=vector_config,
        )
        print(f"Collection '{qdrant_collection_name}' created successfully.")
    else:
        print(f"Collection '{qdrant_collection_name}' already exists.")

    return client


def connect_to_qdrant(qdrant_host: str, qdrant_api_key: str, collection_name: str, embedding_model: OpenAIEmbeddings) -> Qdrant:
    '''
        Connects to a Qdrant collection to create a vector store.

        Args:
            client (QdrantClient): An initialized Qdrant client object.
            collection_name (str): The name of the Qdrant collection to connect to.
            embedding_model (OpenAIEmbeddings): The OpenAI embedding model to use (e.g., "text-embedding-ada-002").

        Returns:
            Qdrant: A Qdrant vector store instance connected to the specified collection.
    '''
    
    client = initial_qdrant_database(qdrant_host=qdrant_host, qdrant_api_key=qdrant_api_key, qdrant_collection_name=collection_name)
    
    vectorstore = Qdrant(
        client= client,
        collection_name= collection_name,
        embeddings=embedding_model,
    )

    return vectorstore

# @retry(tries=100, delay=10)
def seed_qdrant(qdrant_host:str, qdrant_api_key: str, collection_name: str, embedding_model: OpenAIEmbeddings, filename: str, directory: str) -> None:
    """
        Seeds a Qdrant vector store with data from a local file.

        This function loads data from a specified local JSON file, transforms it into
        Langchain Document objects, generates unique IDs for each document, connects to a
        Qdrant database, and adds the documents to the specified collection.

        Args:
            qdrant_host (str): The host address of the Qdrant database.
            qdrant_api_key (str): The API key for authenticating with the Qdrant database.
            collection_name (str): The name of the Qdrant collection to seed.
            embedding_model (OpenAIEmbeddings): The OpenAI embedding model to use (e.g., "text-embedding-ada-002").
            filename (str): The name of the local JSON file containing the data.
            directory (str): The directory where the local JSON file is located.

        Returns:
            None
    """

    # load data
    local_data, doc_name = load_data_from_local(filename=filename, directory=directory)
    LOGGER.info("Loading data successfully!")

    # Convert data to document list
    documents = [
        Document(
            page_content="passage: " + (doc.get('page_content') or ''),
            metadata={
                'source': doc['metadata'].get('source') or '',
                'content_type': doc['metadata'].get('content_type') or 'text/plain',
                'title': doc['metadata'].get('title') or '',
                'language': doc['metadata'].get('language') or '',
                'doc_name': doc_name,
            }
        )
        for doc in local_data
    ]


    # Create unique id for each document 
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vectorstore = connect_to_qdrant(
                                    qdrant_host=qdrant_host,
                                    qdrant_api_key=qdrant_api_key,
                                    collection_name=collection_name,
                                    embedding_model=embedding_model)
    LOGGER.info("Connecting data cloud successfully!")

    vectorstore.add_documents(documents=documents, ids=uuids)
    LOGGER.info("Add documents successfully!")

if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    collection_name = os.getenv("QDRANT_COLLECTION", "sict_documents")

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_directory = str(project_root / "data")

    print("\nStarting Qdrant database seeding...")
    print(f"   Qdrant Host: {qdrant_host}")
    print(f"   Collection: {collection_name}")

    embedding_model = get_e5_embedding_model()

    seed_qdrant(
        qdrant_host=qdrant_host,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        embedding_model=embedding_model,
        filename="sict_database.json",
        directory=data_directory
    )

    print("\n✅ Database seeded successfully!")