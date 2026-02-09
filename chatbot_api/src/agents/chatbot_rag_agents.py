import os
import logging
import traceback
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Qdrant / Embeddings Imports
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

# Custom Imports (Đảm bảo các file này tồn tại)
from .tools.chatbot_retriever_tool import build_optimized_rag_pipeline, BM25IndexManager
from .prompt.system_prompt import system

# --- CẤU HÌNH LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Lumiya-Core")

load_dotenv()

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
QDRANT_HOST = os.getenv('QDRANT_HOST', 'http://localhost:6333')
QDRANT_COLLECTION = os.getenv('COLLECTION_NAME') # Quan trọng: Phải khớp với lúc nạp dữ liệu

DB_CONFIG = {
    "host": os.getenv('HOST'),
    "user": 'root',
    "password": os.getenv('MYSQL_ROOT_PASSWORD'),
    "database": os.getenv('MYSQL_DATABASE'),
}
logger.info(f"-------------------->Thông tin kết nối mysql: {DB_CONFIG}")

# --- 1. KHỞI TẠO COMPONENTS ---
def initialize_components():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing!")
    
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    
    # Kết nối Qdrant
    client = QdrantClient(url=QDRANT_HOST, api_key=os.getenv('QDRANT_API_KEY'))
    
    # Init Vector Store
    # Lưu ý: Nếu Collection chưa tồn tại, dòng này có thể gây lỗi. 
    # Đảm bảo bạn đã chạy main.py để tạo collection trước.
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embedding_model,
    )
    
    chat_model = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    return embedding_model, vector_store, chat_model

# Chạy khởi tạo
embedding_model, vector_store, chat_model = initialize_components()

# --- 2. XÂY DỰNG RETRIEVER PIPELINE & TOOL ---
tools = [] # Mặc định là rỗng

if vector_store:
    try:
        bm25_path = os.getenv("BM25_INDEX_PATH", "bm25_index.pkl")
        force_rebuild = os.getenv("BM25_FORCE_REBUILD", "false").lower() in ("true", "1", "t")
        
        # Logic 1: Lấy documents từ Qdrant để build BM25 (nếu cần)
        documents_for_bm25 = None
        
        # Chỉ lấy dữ liệu nếu cần rebuild hoặc file chưa tồn tại
        if force_rebuild or not os.path.exists(bm25_path):
            logger.info("⚠️ Đang tải dữ liệu từ Qdrant để build BM25 Index (Lần đầu hoặc Force Rebuild)...")
            limit_k = int(os.getenv("BM25_CORPUS_K", "5000"))
            
            # Scroll lấy dữ liệu thô
            response_scroll, _ = vector_store.client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=limit_k,
                with_payload=True
            )
            
            # Chuyển đổi sang Document object
            documents_for_bm25 = [
                Document(page_content=p.payload.get("page_content", ""), metadata=p.payload.get("metadata", {}))
                for p in response_scroll if p.payload and "page_content" in p.payload
            ]
            logger.info(f"Đã tải {len(documents_for_bm25)} documents cho BM25.")

        # Logic 2: Build Pipeline (Hybrid Search + Rerank)
        # Lưu ý: Hàm build_optimized_rag_pipeline bên file kia phải chấp nhận tham số bm25_manager
        retriever = build_optimized_rag_pipeline(
            vector_store=vector_store,
            bm25_manager=BM25IndexManager(bm25_path), 
            documents_for_bm25=documents_for_bm25,
            force_rebuild_bm25=force_rebuild,
        )

        # Logic 3: Định nghĩa Tool (Có Try/Catch an toàn)
        @lc_tool("Search_HaUI_Info")
        def search_haui_info(query: str) -> str:
            """Tra cứu thông tin chính thức về HaUI (Nhân sự, học phí, lịch thi, tin tức, quy chế...). 
            Cần sử dụng công cụ này trước khi kết luận không có dữ liệu."""
            
            logger.info(f"[START] Đang truy xuất dữ liệu cho query: '{query}'")
            try:
                results = retriever.invoke(query)
            except Exception as e:
                logger.error(f" [ERROR] Lỗi nghiêm trọng khi gọi Retriever!")
                return f"Hệ thống gặp lỗi kỹ thuật khi tra cứu: {str(e)}"

            if not results:
                logger.warning(f" [WARN] Không tìm thấy dữ liệu cho: {query}")
                return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            
            logger.info(f"[SUCCESS] Tìm thấy {len(results)} tài liệu.")
            first_doc_snippet = results[0].page_content[:100].replace('\n', ' ')
            logger.info(f"Preview: {first_doc_snippet}...")
            
            return "\n\n".join([f"Nguồn: {d.page_content}" for d in results])

        # --- QUAN TRỌNG: Đăng ký tool vào list ---
        tools = [search_haui_info]
        logger.info(" Đã khởi tạo Tool: Search_HaUI_Info")

    except Exception as e:
        logger.error(f"Lỗi khởi tạo Retriever/Tool: {e}")
        logger.error(traceback.format_exc())
        # Nếu lỗi, tools vẫn là [] để tránh sập app, nhưng Agent sẽ không tìm kiếm được.


# --- 3. LANGGRAPH LOGIC ---
if chat_model:
    # Bind tools vào model
    model_with_tools = chat_model.bind_tools(tools)

    def call_model(state: MessagesState):
        msgs = state["messages"]
        
        # System Prompt
        sys_msg = SystemMessage(content=system)
        
        # Clean up old system messages
        filtered_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]
        input_msgs = [sys_msg] + filtered_msgs

        # Gọi LLM
        response = model_with_tools.invoke(input_msgs)
        
        # Log hành vi Agent
        if response.tool_calls:
            logger.info(f"🛠️ Agent quyết định gọi Tool: {response.tool_calls[0]['name']}")
        else:
            logger.info("🧠 Agent phản hồi trực tiếp (Không dùng Tool).")
            
        return {"messages": [response]}

    # Graph Setup
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    # --- ADAPTER (Giữ nguyên để tương thích API cũ) ---
    class GraphAgentExecutorAdapter:
        def __init__(self, graph):
            self.graph = graph

        def invoke(self, inputs: dict):
            # Chuyển đổi chat_history từ dict sang object LangChain
            msg_list = []
            for m in inputs.get("chat_history", []):
                if m.get("type") == "user":
                    msg_list.append(HumanMessage(content=m.get("content")))
                else:
                    msg_list.append(AIMessage(content=m.get("content")))
            
            if inputs.get("input"):
                msg_list.append(HumanMessage(content=inputs.get("input")))

            result = self.graph.invoke({"messages": msg_list})
            # Lấy tin nhắn AI cuối cùng
            final_ai_msg = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
            return {"output": final_ai_msg.content}

    chatbot_agent_executor = GraphAgentExecutorAdapter(app)
else:
    logger.critical("Không thể khởi tạo Agent do thiếu Chat Model!")
    chatbot_agent_executor = None