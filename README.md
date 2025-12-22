# SICT Chatbot

Hệ thống chatbot cung cấp thông tin về Trường Công nghệ Thông tin và Truyền thông (SICT – HaUI), triển khai dưới dạng Agent dùng RAG (Agentic RAG). Backend là FastAPI, lưu lịch sử hội thoại vào MySQL và truy xuất ngữ cảnh từ Qdrant (vector DB) kết hợp BM25.

## Tổng Quan
- **Kiểu hệ thống:** Agent dùng RAG (Agentic RAG).
- **Mức độ agency (theo bảng xếp hạng):** "Tool call" – LLM quyết định thực thi công cụ (không phải router, không multi-step, không multi-agent).
- **Phạm vi:** Trả lời các câu hỏi trong miền SICT, luôn ưu tiên dùng công cụ `Search` để lấy ngữ cảnh từ cơ sở dữ liệu vector.

## Kiến Trúc
- **`chatbot_api` (FastAPI):**
	- Endpoint `POST /chatbot-rag-agent` nhận `user_id` và `message`, gọi Agent để tạo phản hồi.
	- Lưu/đọc lịch sử chat vào **MySQL** (`agents/db/chat_history.py`).
	- Agent tạo bằng **LangChain OpenAI Functions Agent** (`create_openai_functions_agent`) + **AgentExecutor**.
- **RAG Tool `Search`:**
	- Retriever ensemble: **Qdrant similarity** + **BM25** (`agents/tools/chatbot_retriever_tool.py`).
	- Nếu Qdrant chưa có dữ liệu, hệ thống cảnh báo và có fallback an toàn.
- **Qdrant:** Lưu vector các tài liệu tin tức/thông báo của SICT.
- **Frontend:** `chatbot_frontend` (Streamlit) kết nối API.

Thư mục chính:
- `chatbot_api/` – Backend FastAPI + Agent + models/tools/prompt.
- `chatbot_frontend/` – Ứng dụng giao diện (Streamlit).
- `qdrant_database/` – Script seed dữ liệu vào Qdrant.
- `qdrant_storage/` – Volume dữ liệu Qdrant.
- `data/` – Nguồn dữ liệu thô.

## Biến Môi Trường (`.env`)
Tạo file `.env` tại root dự án với các biến (ví dụ):

```
# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4

# MySQL (dùng trong Agent)
HOST=mysql
USER=root
PASSWORD=your_mysql_root_password
NAME=sict_chatbot

# Qdrant
QDRANT_HOST=http://qdrant:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=sict_documents
```

Ghi chú:
- Khi chạy bằng Docker Compose, đặt `HOST=mysql` và `QDRANT_HOST=http://qdrant:6333` (theo tên service nội bộ).
- `OPENAI_MODEL` mặc định trong code là `gpt-4`.

## Chạy Bằng Docker Compose (Khuyến nghị)
1) Build image cho API và Frontend:

```bash
docker build -t chatbot_api:latest ./chatbot_api
docker build -t chatbot_frontend:latest ./chatbot_frontend
```

2) Khởi động toàn bộ stack:

```bash
docker compose up -d
```

3) Kiểm tra dịch vụ:
- API: `http://localhost:8080/` (trả về `{"status":"running"}`)
- Frontend: `http://localhost:8501/`
- Qdrant: `http://localhost:6333/`

4) Seed dữ liệu vào Qdrant (nếu bộ sưu tập chưa tồn tại):

```bash
# Chạy trực tiếp trên máy (yêu cầu Python và dependencies)
python qdrant_database/src/seed_data.py

# Hoặc chạy trong container API nếu đã có Python và requirements
docker exec -it chatbot_api_container python qdrant_database/src/seed_data.py
```

## Chạy Cục Bộ (không dùng Docker)
```bash
# Backend API
python3 -m venv .venv
source .venv/bin/activate
pip install -r chatbot_api/requirements.txt
export OPENAI_API_KEY=your_openai_key
export OPENAI_MODEL=gpt-4
export HOST=localhost
export USER=root
export PASSWORD=your_mysql_root_password
export NAME=sict_chatbot
export QDRANT_HOST=http://localhost:6333
export QDRANT_COLLECTION=sict_documents

uvicorn chatbot_api.src.main:app --host 0.0.0.0 --port 8080 --reload
```

Frontend (tùy chọn):
```bash
python3 -m venv .venv-frontend
source .venv-frontend/bin/activate
pip install -r chatbot_frontend/requirements.txt
python chatbot_frontend/src/main.py
```

## API Docs Nhanh
- `GET /` → kiểm tra tình trạng dịch vụ.
- `POST /chatbot-rag-agent`
	- Request JSON:
		```json
		{ "user_id": "u123", "message": "Học phí ngành CNTT?" }
		```
	- Response JSON:
		```json
		{ "user_id": "u123", "message": "Học phí ngành CNTT?", "response": "...", "error": null }
		```

Ví dụ gọi nhanh bằng `curl`:
```bash
curl -X POST http://localhost:8080/chatbot-rag-agent \
	-H "Content-Type: application/json" \
	-d '{"user_id":"demo","message":"Sắp tới có sự kiện gì cho tân sinh viên?"}'
```

## Ràng Buộc & An Toàn
- Prompt hệ thống giới hạn nội dung trong miền SICT, luôn ưu tiên dùng công cụ `Search`.
- Nếu không tìm thấy ngữ cảnh, hệ thống trả lời trung thực và hướng dẫn liên hệ phòng ban.

## Lưu Ý Vận Hành
- Nếu log báo: "Collection 'sict_documents' not found", hãy chạy script seed để tạo dữ liệu.
- Khi Qdrant hoặc Embeddings chưa khởi tạo, Agent có thể đáp không kèm ngữ cảnh; cân nhắc fail-closed nếu muốn tránh suy đoán.
- Lịch sử chat lưu theo `session_id`; có hỗ trợ reset trong lớp `MySQLChatMessageHistory`.