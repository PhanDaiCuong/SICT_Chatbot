from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333", api_key=None)

collection_name = "sict_news" # Tên trong file config của bạn

# Xóa collection nếu nó tồn tại
client.delete_collection(collection_name=collection_name)
print(f"Đã xóa collection '{collection_name}' thành công.")