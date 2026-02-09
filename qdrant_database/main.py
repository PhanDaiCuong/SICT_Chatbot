import os
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from utils.config import PATH_MAPPING, CLASSIFICATION_SETS

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger("LumiYa-Seeding")

# ==========================================
# 1. HÀM XỬ LÝ METADATA & LÀM SẠCH TÊN TRƯỜNG
# ==========================================
def clean_school_name(name: str) -> str:
    """Xóa hậu tố _corpus và chuẩn hóa tên trường."""
    if not name: return None
    return name.lower().replace("_corpus", "").strip()

def generate_optimized_metadata(file_path_str):
    path = Path(file_path_str)
    parts = path.parts
    metadata = {"school": None, "topics": []}
    context_parts = []

    for part in parts:
        part_key = part.lower()
        if part_key in PATH_MAPPING:
            human_name = PATH_MAPPING[part_key]
            if part_key in CLASSIFICATION_SETS.get("SCHOOLS", []):
                clean_name = clean_school_name(part_key)
                metadata["school"] = clean_name
                context_parts.append(f"Trường: {human_name}")
            else:
                metadata["topics"].append(human_name)

    metadata["topics"] = ", ".join(metadata["topics"]) if metadata["topics"] else "Chung"
    context_prefix = " - ".join(context_parts) if context_parts else "HaUI"
    return context_prefix, metadata

# ==========================================
# 2. HÀM KHỞI TẠO COLLECTION (NẾU CHƯA CÓ)
# ==========================================
def ensure_collection_exists(client: QdrantClient, collection_name: str):
    """Kiểm tra và tạo collection với cấu hình chuẩn nếu chưa tồn tại."""
    try:
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            LOGGER.info(f"✨ Đang tạo collection mới: '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536, # Kích thước của text-embedding-3-small
                    distance=models.Distance.COSINE
                )
            )
            LOGGER.info(f"Đã tạo thành công collection '{collection_name}'.")
        else:
            LOGGER.info(f"ℹ Collection '{collection_name}' đã tồn tại.")
    except Exception as e:
        LOGGER.error(f"Lỗi khi khởi tạo Collection: {e}")
        sys.exit(1)

# ==========================================
# 3. HÀM CHUNK & ĐỊNH DẠNG PAGE_CONTENT
# ==========================================
def process_file_optimized(file_path: Path, text_splitter) -> list[Document]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0: data = data[0]

        context_prefix, meta = generate_optimized_metadata(str(file_path))
        meta["title"] = data.get('title', 'Thông báo')
        meta["url"] = data.get("url", "")

        # Ép nội dung vào cấu trúc chuẩn để Embedding hiểu ngữ cảnh
        full_text_content = f"[{context_prefix}] {meta['title']}: {data.get('content', '')}"

        # LangChain sẽ tự động gán text này vào 'page_content' của Document
        return text_splitter.create_documents([full_text_content], metadatas=[meta])
    except Exception as e:
        LOGGER.error(f"Lỗi file {file_path.name}: {e}")
        return []

# ==========================================
# 4. RUN PIPELINE
# ==========================================
def run_seeding_pipeline():
    load_dotenv()
    Q_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
    Q_COLLECTION = os.getenv("COLLECTION_NAME")
    OAI_KEY = os.getenv("OPENAI_API_KEY")
    DATA_DIR = os.getenv("DATA_DIR")

    # 1. Init Components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OAI_KEY)
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_amount=95)
    client = QdrantClient(url=Q_HOST, api_key=os.getenv("QDRANT_API_KEY"))

    # 2. Đảm bảo Collection tồn tại trước khi nạp
    ensure_collection_exists(client, Q_COLLECTION)

    # 3. Khởi tạo VectorStore với key page_content
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=Q_COLLECTION,
        embedding=embeddings,
        content_payload_key="page_content"
    )

    # 4. Xử lý file
    files = list(Path(DATA_DIR).rglob("*.json"))
    LOGGER.info(f"Bắt đầu nạp {len(files)} file...")

    batch_docs = []
    for f_path in tqdm(files, desc="Đang nạp dữ liệu"):
        chunks = process_file_optimized(f_path, text_splitter)
        batch_docs.extend(chunks)

        if len(batch_docs) >= 100:
            vectorstore.add_documents(batch_docs)
            batch_docs = []

    if batch_docs:
        vectorstore.add_documents(batch_docs)

    LOGGER.info("Hoàn tất seeding dữ liệu tối ưu!")

if __name__ == "__main__":
    run_seeding_pipeline()