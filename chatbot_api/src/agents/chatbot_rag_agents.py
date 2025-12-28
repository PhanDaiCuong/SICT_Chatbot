import os
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from .tools.chatbot_retriever_tool import build_optimized_rag_pipeline, BM25IndexManager
from .prompt.system_prompt import system
from .base.llm_model import get_llm



logger = logging.getLogger(__name__)

load_dotenv()
hf_llm_model = os.getenv('HF_LLM_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')
host = os.getenv('HOST')
user = os.getenv('USER')
password = os.getenv('PASSWORD')
database = os.getenv('NAME')
qdrant_host = os.getenv('QDRANT_HOST', 'http://localhost:6333')
qdrant_api_key = os.getenv('QDRANT_API_KEY', '')
qdrant_collection = os.getenv('QDRANT_COLLECTION', 'sict_documents')
hf_embedding_model = os.getenv('HF_EMBEDDING_MODEL', 'intfloat/multilingual-e5-base')

# Set up information for account of Mysql
DB_CONFIG = {
    "host": host,
    "user": user,
    "password": password,
    "database": database,
}

# Initialize OpenAI Embeddings for RAG
try:
    logger.info("Initializing HuggingFace embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=hf_embedding_model)
    logger.info("Embedding model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embedding model: {str(e)}")
    embedding_model = None

# Initialize Qdrant Vector Store
try:
    logger.info(f"Connecting to Qdrant at {qdrant_host}...")
    qdrant_client = QdrantClient(
        url=qdrant_host,
        api_key=qdrant_api_key if qdrant_api_key else None,
    )
    
    # Check if collection exists
    collections = qdrant_client.get_collections().collections
    existing_collections = [col.name for col in collections]
    
    if qdrant_collection not in existing_collections:
        logger.warning(f"Collection '{qdrant_collection}' not found in Qdrant")
        logger.warning("    Run: python qdrant_database/src/seed_data.py to populate the database")
        vector_store = None
    else:
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name=qdrant_collection,
            embeddings=embedding_model,
        )
        logger.info(f"Connected to Qdrant collection: {qdrant_collection}")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant: {str(e)}")
    logger.error("   Make sure Qdrant server is running")
    vector_store = None

# Create tool for agent
if vector_store and embedding_model:
    try:
        logger.info("Creating optimized retriever pipeline from vector store...")
        bm25_index_path = os.getenv("BM25_INDEX_PATH", "bm25_index.pkl")
        bm25_force_rebuild = os.getenv("BM25_FORCE_REBUILD", "false").lower() in ("true", "1", "t")
        bm25_corpus_k = int(os.getenv("BM25_CORPUS_K", "2000"))

        # Snapshot corpus for BM25 index
        documents_for_bm25 = vector_store.similarity_search("", k=bm25_corpus_k)
        bm25_manager = BM25IndexManager(index_path=bm25_index_path)

        retriever = build_optimized_rag_pipeline(
            vector_store=vector_store,
            bm25_manager=bm25_manager,
            documents_for_bm25=documents_for_bm25,
            force_rebuild_bm25=bm25_force_rebuild,
        )
        tool = create_retriever_tool(
            retriever=retriever,
            name="Search",
            description="Search the information of SICT news, events, training regulations, and announcements from the vector database."
        )
        tools = [tool]
        logger.info("Retriever tool created successfully")
    except Exception as e:
        logger.error(f"Failed to create retriever tool: {str(e)}")
        tools = []
else:
    logger.warning("Vector store not initialized - Chatbot will respond without RAG context")
    tools = []

try:
    logger.info(f"Loading HuggingFace LLM: {hf_llm_model}")
    chat_model = get_llm(model_name=hf_llm_model, max_new_tokens=int(os.getenv('MAX_NEW_TOKENS', '512')))
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to load HF LLM: {e}")
    raise

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent for system (ReAct agent works with generic LLMs)
chatbot_agent = create_react_agent(
    llm=chat_model,
    tools=tools,
    prompt=prompt,
)

chatbot_agent_executor = AgentExecutor(
    agent=chatbot_agent,
    tools=tools,
    verbose=True,
)
