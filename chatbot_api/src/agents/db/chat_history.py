import os
import mysql.connector
import logging
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels



logging.basicConfig(level=logging.INFO)

def create_chat_history_table(DB_CONFIG: dict) -> None:
    """
        Creates the `chat_history` table in the MySQL database if it doesn't already exist.
        The table stores chat messages with session ID, message type, and content.
    """
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(255) NOT NULL,
        message_type ENUM('user', 'ai') NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_table_query)
    connection.commit()
    cursor.close()
    connection.close()
    logging.info("Chat history table created (or already exists).")


class MySQLChatMessageHistory:
    """
        A class to manage chat history in MySQL, allowing message storage and retrieval.
    """
    def __init__(self, session_id, DB_CONFIG: dict):
        self.session_id = session_id
        self.DB_CONFIG = DB_CONFIG

    
    def add_message(self, message_type: str, content: str) -> None:
        """
            Adds a message to the database.

            Args:
                message_type (str): Type of the message ('user' or 'ai').
                content (str): The message content.
        """
        connection = mysql.connector.connect(**self.DB_CONFIG)
        cursor = connection.cursor()
        insert_query="""
        INSERT INTO chat_history (session_id, message_type, content)
        VALUES (%s, %s, %s);
        """

        cursor.execute(insert_query, (self.session_id, message_type, content,))
        connection.commit()
        logging.info("Add message successfully!")
        cursor.close()
        connection.close()

    
    def load_messages(self) -> list:
        """
            Retrieves all messages for the session from the database.

            Returns:
                list: List of messages as dictionaries.
        """
        connection = mysql.connector.connect(**self.DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        select_query =  """ SELECT message_type, content 
                            FROM chat_history
                            WHERE session_id = %s ORDER BY created_at;
                        """
        cursor.execute(select_query, (self.session_id,))
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        logging.info("Loading message successfully!")
        
        return [{"type": row["message_type"], "content": row["content"]} for row in rows]
    

    def reset_history(self) -> None:
        """
            Deletes all messages for the session from the database.
        """
        connection = mysql.connector.connect(**self.DB_CONFIG)
        cursor = connection.cursor()
        delete_query = "DELETE FROM chat_history WHERE session_id = %s;"
        cursor.execute(delete_query, (self.session_id,))
        connection.commit()
        logging.info("Reseting message successfully!")
        cursor.close()
        connection.close()


class ChatHistoryVectorStore:
    """
        Store and retrieve chat history as vectors in Qdrant, scoped by session_id.

        - Adds each message as a Document(page_content=content, metadata={session_id, type}).
        - Retrieves only the most relevant past messages for a given query using similarity search
          with a filter on session_id.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embeddings: HuggingFaceEmbeddings,
        collection_name: str = "sict_chat_history",
    ):
        self.collection_name = collection_name
        self.vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=embeddings,
        )
        logging.info(f"Chat history vector store ready: collection '{collection_name}'")

    def add_message(self, session_id: str, message_type: str, content: str) -> None:
        doc = Document(page_content=content, metadata={"session_id": session_id, "type": message_type})
        self.vector_store.add_documents([doc])
        logging.info("Added chat message to Qdrant vector store")

    def search_relevant(self, session_id: str, query: str, k: int = 6) -> List[dict]:
        """
            Return top-k relevant messages in the session.
        """
        filter_ = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="session_id",
                    match=qmodels.MatchValue(value=session_id)
                )
            ]
        )
        docs = self.vector_store.similarity_search(query, k=k, filter=filter_)
        return [{"type": d.metadata.get("type", "user"), "content": d.page_content} for d in docs]


_chat_history_vectors = None

def get_chat_history_vector_store() -> ChatHistoryVectorStore:
    """
        Initialize and cache a ChatHistoryVectorStore using environment variables.
        Env vars:
        - HF_EMBEDDING_MODEL (default: intfloat/multilingual-e5-base)
        - QDRANT_HOST (default: http://localhost:6333)
        - QDRANT_API_KEY (optional)
        - CHAT_HISTORY_COLLECTION (default: sict_chat_history)
    """
    global _chat_history_vectors
    if _chat_history_vectors is not None:
        return _chat_history_vectors

    qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    collection = os.getenv("CHAT_HISTORY_COLLECTION", "sict_chat_history")
    hf_embedding_model = os.getenv("HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

    embeddings = HuggingFaceEmbeddings(model_name=hf_embedding_model)
    client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key if qdrant_api_key else None)
    _chat_history_vectors = ChatHistoryVectorStore(client, embeddings, collection)
    return _chat_history_vectors

