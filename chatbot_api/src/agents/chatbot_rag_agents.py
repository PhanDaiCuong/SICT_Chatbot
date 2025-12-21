import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import (
    create_openai_functions_agent,
    AgentExecutor,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from .tools.chatbot_retriever_tool import get_retriever
from .prompt.system_prompt import system

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()
openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
openai_api_key = os.getenv("OPENAI_API_KEY")
host = os.getenv('HOST')
user = os.getenv('USER')
password = os.getenv('PASSWORD')
database = os.getenv('NAME')
qdrant_host = os.getenv('QDRANT_HOST', 'http://localhost:6333')
qdrant_api_key = os.getenv('QDRANT_API_KEY', '')
qdrant_collection = os.getenv('QDRANT_COLLECTION', 'sict_documents')

# Set up information for account of Mysql
DB_CONFIG = {
    "host": host,
    "user": user,
    "password": password,
    "database": database,
}

# Initialize OpenAI Embeddings for RAG
try:
    logger.info("Initializing OpenAI embeddings model...")
    embedding_model = OpenAIEmbeddings(
        api_key=openai_api_key,
        model="text-embedding-ada-002"
    )
    logger.info("✅ Embedding model initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize embedding model: {str(e)}")
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
        logger.warning(f"⚠️  Collection '{qdrant_collection}' not found in Qdrant")
        logger.warning("    Run: python qdrant_database/src/seed_data.py to populate the database")
        vector_store = None
    else:
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name=qdrant_collection,
            embeddings=embedding_model,
        )
        logger.info(f"✅ Connected to Qdrant collection: {qdrant_collection}")
except Exception as e:
    logger.error(f"❌ Failed to initialize Qdrant: {str(e)}")
    logger.error("   Make sure Qdrant server is running")
    vector_store = None

# Create tool for agent
if vector_store and embedding_model:
    try:
        logger.info("Creating retriever tool from vector store...")
        retriever = get_retriever(vector_store, qdrant_collection)
        tool = create_retriever_tool(
            retriever=retriever,
            name="Search",
            description="Search the information of SICT news, events, training regulations, and announcements from the vector database."
        )
        tools = [tool]
        logger.info("✅ Retriever tool created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create retriever tool: {str(e)}")
        tools = []
else:
    logger.warning("⚠️  Vector store not initialized - Chatbot will respond without RAG context")
    tools = []

chat_model = ChatOpenAI(
    model = openai_model,
    temperature=0,
    streaming=True,
    api_key=openai_api_key,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent for system
chatbot_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=prompt,
    tools=tools
)
chatbot_agent_executor = AgentExecutor(
    agent=chatbot_agent,
    tools=tools,
    verbose=True,
)
