import os
import logging
import mysql.connector
from typing import List, Dict, Optional

# --- 1. MODERN IMPORTS (LangChain 0.3) ---
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models as qmodels

# Sử dụng thư viện chuyên biệt (Cần cài: pip install langchain-qdrant langchain-huggingface)
try:
    from langchain_qdrant import QdrantVectorStore
except ImportError:
    # Fallback nếu chưa cài library mới
    from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_chat_history_table(DB_CONFIG: dict) -> None:
    """Tạo bảng MySQL nếu chưa tồn tại (Sử dụng Context Manager an toàn)."""
    logger.info(f"Tham số kết nối trực tiếp đến mysql ------------------- {DB_CONFIG}")
    try:
        with mysql.connector.connect(**DB_CONFIG) as connection:
            with connection.cursor() as cursor:
                create_table_query = """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    message_type ENUM('user', 'ai') NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session (session_id) -- Tối ưu tốc độ tìm kiếm theo session
                );
                """
                cursor.execute(create_table_query)
                connection.commit()
        logger.info("Chat history table check/creation completed.")
    except mysql.connector.Error as err:
        logger.error(f"MySQL Error: {err}")


class MySQLChatMessageHistory:
    """Quản lý lịch sử chat MySQL với cơ chế đóng mở kết nối an toàn."""
    
    def __init__(self, session_id: str, DB_CONFIG: dict):
        self.session_id = session_id
        self.DB_CONFIG = DB_CONFIG

    def add_message(self, message_type: str, content: str) -> None:
        try:
            with mysql.connector.connect(**self.DB_CONFIG) as connection:
                with connection.cursor() as cursor:
                    insert_query = """
                    INSERT INTO chat_history (session_id, message_type, content)
                    VALUES (%s, %s, %s);
                    """
                    cursor.execute(insert_query, (self.session_id, message_type, content))
                    connection.commit()
            # logger.info(f"Saved {message_type} message to MySQL.")
        except mysql.connector.Error as err:
            logger.error(f"Failed to add message to MySQL: {err}")

    def load_messages(self) -> List[Dict[str, str]]:
        messages = []
        try:
            with mysql.connector.connect(**self.DB_CONFIG) as connection:
                # dictionary=True giúp trả về dict thay vì tuple
                with connection.cursor(dictionary=True) as cursor:
                    select_query = """
                        SELECT message_type, content 
                        FROM chat_history
                        WHERE session_id = %s 
                        ORDER BY created_at ASC;
                    """
                    cursor.execute(select_query, (self.session_id,))
                    rows = cursor.fetchall()
                    messages = [{"type": row["message_type"], "content": row["content"]} for row in rows]
            logger.info(f"Loaded {len(messages)} messages from MySQL.")
        except mysql.connector.Error as err:
            logger.error(f"Failed to load messages from MySQL: {err}")
        return messages

    def reset_history(self) -> None:
        try:
            with mysql.connector.connect(**self.DB_CONFIG) as connection:
                with connection.cursor() as cursor:
                    delete_query = "DELETE FROM chat_history WHERE session_id = %s;"
                    cursor.execute(delete_query, (self.session_id,))
                    connection.commit()
            logger.info(f"History reset for session {self.session_id}.")
        except mysql.connector.Error as err:
            logger.error(f"Failed to reset history: {err}")


class ChatHistoryVectorStore:
    """
    Lưu trữ ngữ nghĩa của hội thoại vào Qdrant.
    Giúp Agent nhớ được các chi tiết cũ trong cuộc hội thoại dài.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embeddings: OpenAIEmbeddings,
        collection_name: str = "chat_history",
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        
        # 1. Đảm bảo collection tồn tại trước khi khởi tạo Store
        self._ensure_collection_exists(embeddings)

        # 2. Khởi tạo Vector Store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        logger.info(f"Chat Vector Store ready: {collection_name}")

    def _ensure_collection_exists(self, embeddings):
        """Kiểm tra và tạo collection nếu chưa có"""
        try:
            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)
            
            if not exists:
                logger.info(f"Creating new chat history collection: {self.collection_name}")
                # Lấy kích thước vector từ model embedding (thường là 768 hoặc 384)
                # Hack nhẹ: nhúng thử 1 từ để lấy dimension
                sample_embedding = embeddings.embed_query("test")
                vector_size = len(sample_embedding)
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection: {e}")

    def add_message(self, session_id: str, message_type: str, content: str) -> None:
        """Lưu tin nhắn dưới dạng vector kèm metadata"""
        try:
            doc = Document(
                page_content=content, 
                metadata={"session_id": session_id, "type": message_type}
            )
            self.vector_store.add_documents([doc])
        except Exception as e:
            logger.error(f"Failed to vectorise message: {e}")

    def search_relevant(self, session_id: str, query: str, k: int = 4) -> List[Dict[str, str]]:
        """Tìm các tin nhắn cũ có liên quan đến câu hỏi hiện tại"""
        try:
            # Filter quan trọng: Chỉ tìm trong session_id hiện tại
            session_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.session_id", # Lưu ý: LangChain Qdrant lưu metadata lồng bên trong
                        match=qmodels.MatchValue(value=session_id)
                    )
                ]
            )
            
            docs = self.vector_store.similarity_search(query, k=k, filter=session_filter)
            return [{"type": d.metadata.get("type", "user"), "content": d.page_content} for d in docs]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []


# --- SINGLETON INSTANCE ---
_chat_history_vectors = None

def get_chat_history_vector_store() -> ChatHistoryVectorStore:
    """Singleton Pattern để tránh khởi tạo lại kết nối Qdrant nhiều lần"""
    global _chat_history_vectors
    
    if _chat_history_vectors is not None:
        return _chat_history_vectors

    # Load Env
    qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    # Dedicated chat history collection to avoid conflicts
    collection = os.getenv("CHAT_HISTORY_COLLECTION", "chat_history")
    oai_embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    logger.info("Initializing Global Chat History Vector Store...")
    
    # Init Components
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env for chat history embeddings")
    embeddings = OpenAIEmbeddings(model=oai_embed_model, api_key=openai_api_key)
    client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
    
    _chat_history_vectors = ChatHistoryVectorStore(client, embeddings, collection)
    return _chat_history_vectors