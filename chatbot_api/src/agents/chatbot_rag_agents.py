"""
RAG Agent setup using LangChain + LangGraph with hybrid retrieval tools.

This module wires together:
- OpenAI chat model for response generation.
- Qdrant vector store for semantic search.
- A custom retrieval tool (`Search_HaUI_Info`) built on an optimized pipeline.
- A LangGraph workflow that decides when to call tools.
- A small adapter that exposes a simple `.invoke({...})` API.

Environment Variables
---------------------
Required:
- `OPENAI_API_KEY`: API key for the OpenAI model.

Optional (with defaults):
- `OPENAI_MODEL` (default: `gpt-4o-mini`): Chat model name.
- `EMBEDDING_MODEL` (default: `text-embedding-3-small`): Embedding model.
- `QDRANT_HOST` (default: `http://localhost:6333`): Qdrant endpoint.
- `COLLECTION_NAME`: Name of the Qdrant collection (must exist ahead of time).
- `QDRANT_API_KEY`: If your Qdrant instance requires authentication.
- `BM25_INDEX_PATH` (default: `bm25_index.pkl`): Path to cached BM25 index.
- `BM25_FORCE_REBUILD` (default: `false`): Force rebuild the BM25 index.
- `BM25_CORPUS_K` (default: `5000`): Max payloads to pull when building BM25.

Database (optional, currently logged for visibility):
- `HOST`, `MYSQL_ROOT_PASSWORD`, `MYSQL_DATABASE` (see `DB_CONFIG`).

Quick Start
-----------
1) Ensure Qdrant is running and the collection is created (via your ingestion).
2) Set environment variables (e.g., via `.env`).
3) Import `chatbot_agent_executor` and call `.invoke({"input": "..."})`.

Example
-------
````python
from chatbot_api.src.agents.chatbot_rag_agents import chatbot_agent_executor

result = chatbot_agent_executor.invoke({
    "chat_history": [{"type": "user", "content": "Hello"}],
    "input": "What is the admission deadline?"
})
print(result["output"])  # Final assistant response
````
"""

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
    """Initialize embeddings, Qdrant vector store, and chat model.

    Returns
    -------
    tuple
        A triple `(embedding_model, vector_store, chat_model)`.

    Raises
    ------
    ValueError
        If `OPENAI_API_KEY` is missing.

    Notes
    -----
    - Ensure your Qdrant collection exists; otherwise the vector store
      initialization may fail. Run your ingestion beforehand.
    - `ChatOpenAI` is created with low temperature for deterministic outputs.
    """
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
        retriever = build_optimized_rag_pipeline(
            vector_store=vector_store,
            bm25_manager=BM25IndexManager(bm25_path), 
            documents_for_bm25=documents_for_bm25,
            force_rebuild_bm25=force_rebuild,
        )

        # Logic 3: Định nghĩa Tool (Có Try/Catch an toàn)
        @lc_tool("Search_HaUI_Info")
        def search_haui_info(query: str) -> str:
            """Search official HaUI information via the hybrid RAG retriever.

            Use this tool to fetch authoritative information (staff, tuition,
            schedules, news, regulations, etc.). Prefer calling this before
            concluding that data is unavailable.

            Parameters
            ----------
            query : str
                Natural language query to search in the indexed corpus.

            Returns
            -------
            str
                A newline-separated string of sources concatenating page content.
                If empty or an error occurs, returns a descriptive message.

            Notes
            -----
            - Logs success and a small preview of the first result.
            - Catches exceptions to keep the agent responsive.
            """
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
        """Core node: build input messages, call the LLM, and return response.

        Steps
        -----
        - Prepend the `system` prompt to incoming messages.
        - Invoke the model (with tools bound) and capture its response.
        - Log whether the agent decided to use a tool.

        Parameters
        ----------
        state : MessagesState
            The current message state managed by LangGraph.

        Returns
        -------
        dict
            A dict with a `messages` key containing the latest assistant message.
        """
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
        """Thin adapter exposing a simple `.invoke` API over a LangGraph app.

        Use this to integrate with existing HTTP handlers or CLI code where
        you want to pass a dict containing `chat_history` and `input`, and
        receive a dict with `output`.
        """
        def __init__(self, graph):
            """Store the compiled graph for later invocations.

            Parameters
            ----------
            graph : Any
                The compiled LangGraph application.
            """
            self.graph = graph

        def invoke(self, inputs: dict):
            """Run a single conversational step through the graph.

            Parameters
            ----------
            inputs : dict
                Dict with optional `chat_history` (list of dicts with `type`
                and `content`) and `input` (latest user message).

            Returns
            -------
            dict
                Dict with a single key `output` containing the assistant's
                message text from the last response.

            Examples
            --------
            ````python
            adapter = GraphAgentExecutorAdapter(app)
            resp = adapter.invoke({
                "chat_history": [{"type": "user", "content": "hi"}],
                "input": "Tell me campus news"
            })
            print(resp["output"])  # assistant response
            ````
            """
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