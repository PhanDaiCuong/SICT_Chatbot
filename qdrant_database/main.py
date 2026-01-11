import os
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from utils.config import PATH_MAPPING, CLASSIFICATION_SETS


# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


# ==========================================
# 1. HÀM XỬ LÝ NGỮ CẢNH (CONTEXT LOGIC)
# ==========================================
def generate_dynamic_context(file_path_str):
    """
    Phân tích đường dẫn file để tạo ngữ cảnh tự nhiên và metadata phân loại.
    """
    path = Path(file_path_str)
    parts = path.parts
    
    metadata = {
        "school": None,
        "major": None,
        "department": None,
        "level": None,
        "topics": [],
        "raw_path": str(file_path_str) # Lưu đường dẫn gốc để debug
    }
    
    context_keywords = []

    for part in parts:
        part_key = part.lower()
        
        # Chỉ xử lý nếu key có trong từ điển Mapping
        if part_key in PATH_MAPPING:
            human_text = PATH_MAPPING[part_key]
            
            # --- Logic Phân Loại ---
            if part_key in CLASSIFICATION_SETS["SCHOOLS"]:
                metadata["school"] = part_key
                context_keywords.append(f"Đơn vị: {human_text}")
                
            elif part_key in CLASSIFICATION_SETS["MAJORS"]:
                metadata["major"] = part_key
                context_keywords.append(f"Ngành: {human_text}")
                
            elif part_key in CLASSIFICATION_SETS["DEPARTMENTS"]:
                metadata["department"] = part_key
                context_keywords.append(f"Đơn vị trực thuộc: {human_text}")
                
            elif part_key in CLASSIFICATION_SETS["LEVELS"]:
                metadata["level"] = part_key
                context_keywords.append(f"Hệ đào tạo: {human_text}")
                
            else:
                metadata["topics"].append(part_key)
                context_keywords.append(f"Mục: {human_text}")

    # Tạo câu Context để tiêm vào nội dung
    if context_keywords:
        full_context_str = " - ".join(context_keywords) + "."
    else:
        full_context_str = "Thông tin chung Đại học Công nghiệp Hà Nội."
        
    # Làm gọn metadata topics
    metadata["topics"] = ", ".join(metadata["topics"]) if metadata["topics"] else None

    return full_context_str, metadata

# ==========================================
# 2. HÀM LOAD & CHUNK DỮ LIỆU
# ==========================================
def process_file_semantic(file_path: Path, text_splitter) -> list[Document]:
    """
    Đọc 1 file, tiêm ngữ cảnh, và chia nhỏ bằng Semantic Chunking.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Kiểm tra format JSON cơ bản
        # Giả sử file json chứa 1 object {title, abstract, content...}
        # Nếu json là list các bài viết, cần vòng lặp for ở đây.
        if isinstance(data, list):
            # Nếu file json chứa list, ta xử lý bài đầu tiên hoặc loop (tùy cấu trúc data của bạn)
            # Ở đây giả định 1 file = 1 bài viết để tối ưu ngữ cảnh folder
            data = data[0] if data else {}

        # 1. Lấy ngữ cảnh từ đường dẫn
        context_str, path_metadata = generate_dynamic_context(str(file_path))

        # 2. Chuẩn bị nội dung thô (Raw Text) với Context Injection
        # Đưa Context lên đầu để Semantic Model hiểu ngay ngữ cảnh
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        content = data.get('content', '')
        
        # Format text để chunking
        raw_text = f"{context_str}\n\nTiêu đề: {title}\n\nTóm tắt: {abstract}\n\nNội dung: {content}"

        # 3. Chuẩn bị Metadata gốc từ file
        file_metadata = {
            "title": title,
            "url": data.get("url", ""),
            "id": data.get("id", str(uuid4())),
            "image_url": data.get("images", [{}])[0].get("original_url") if data.get("images") else None
        }
        
        # Merge metadata từ Path và metadata từ File
        final_metadata = {**path_metadata, **file_metadata}

        # 4. Thực hiện Semantic Chunking
        docs = text_splitter.create_documents([raw_text], metadatas=[final_metadata])
        
        return docs

    except Exception as e:
        LOGGER.error(f"Failed to process file {file_path}: {e}")
        return []

# ==========================================
# 3. HÀM QUẢN LÝ KẾT NỐI QDRANT
# ==========================================
def init_qdrant_collection(client: QdrantClient, collection_name: str):
    """Đảm bảo Collection tồn tại với config đúng."""
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        LOGGER.info(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,  
                distance=models.Distance.COSINE
            )
        )
    else:
        LOGGER.info(f"Collection '{collection_name}' already exists.")

# ==========================================
# 4. MAIN SEEDING FUNCTION (BATCH PROCESSING)
# ==========================================
def seed_qdrant_recursive(
    root_dir: str, 
    qdrant_host: str, 
    qdrant_api_key: str, 
    collection_name: str, 
    openai_api_key: str
):
    # 1. Setup Models
    embedding_model = OpenAIEmbeddings(
        api_key=openai_api_key,
        model="text-embedding-3-small" # Khuyên dùng model này thay vì ada-002 (rẻ hơn & tốt hơn)
    )

    # Semantic Chunker setup
    # breakpoint_threshold_type="percentile": Cắt dựa trên sự thay đổi ngữ nghĩa đột ngột
    text_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90 # Ngưỡng nhạy (90-95 là tốt cho văn bản tin tức)
    )

    # 2. Setup Qdrant Client
    client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)
    init_qdrant_collection(client, collection_name)
    
    # Kết nối VectorStore
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_model
    )

    # 3. Quét toàn bộ file JSON (Recursive)
    root_path = Path(root_dir)
    json_files = list(root_path.rglob("*.json"))
    LOGGER.info(f"Found {len(json_files)} JSON files in {root_dir}")

    # 4. Xử lý & Upload theo Batch (Để tiết kiệm RAM)
    BATCH_SIZE = 50 # Số lượng chunks sẽ upload 1 lần
    chunk_buffer = []
    
    # Dùng tqdm để hiện thanh loading
    for file_path in tqdm(json_files, desc="Processing Files"):
        
        # Xử lý từng file -> ra nhiều chunks
        chunks = process_file_semantic(file_path, text_splitter)
        
        # Thêm vào bộ đệm
        chunk_buffer.extend(chunks)

        # Nếu bộ đệm đầy thì đẩy lên Qdrant
        if len(chunk_buffer) >= BATCH_SIZE:
            vectorstore.add_documents(chunk_buffer)
            chunk_buffer = [] # Clear buffer

    # Đẩy nốt số còn dư
    if chunk_buffer:
        vectorstore.add_documents(chunk_buffer)

    LOGGER.info("Seeding completed successfully!")

# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    load_dotenv()
    
    # Load Env Variables
    QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sict_documents_semantic")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Đường dẫn data gốc (Nơi chứa folder sict_corpus, seee_corpus...)
    # Lưu ý: Trỏ vào folder cha chứa các corpus
    current_dir = Path(__file__).parent
    DATA_DIRECTORY = os.getenv("DATA_DIR", str(current_dir.parent.parent / "data"))

    if not OPENAI_API_KEY:
        LOGGER.error("OPENAI_API_KEY is missing!")
        sys.exit(1)

    if not os.path.exists(DATA_DIRECTORY):
        LOGGER.error(f"Data directory not found: {DATA_DIRECTORY}")
        sys.exit(1)

    print("\n🚀 STARTING SEMANTIC SEEDING PIPELINE 🚀")
    print(f"Target: {QDRANT_HOST} | Collection: {QDRANT_COLLECTION}")
    print(f"Scanning Data: {DATA_DIRECTORY}")
    
    try:
        seed_qdrant_recursive(
            root_dir=DATA_DIRECTORY,
            qdrant_host=QDRANT_HOST,
            qdrant_api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION,
            openai_api_key=OPENAI_API_KEY
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        LOGGER.critical(f"Fatal Error: {e}")