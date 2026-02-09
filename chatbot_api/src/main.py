from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
import uvicorn

from .agents.db.chat_history import (
    MySQLChatMessageHistory,
    create_chat_history_table
)
from .models.chatbot_rag_query import ChatRequest, ChatResponse
from .agents.chatbot_rag_agents import chatbot_agent_executor, DB_CONFIG


# -------------------------
# Logging config
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("Lumiya-Chatbot")


# -------------------------
# Lifespan (startup / shutdown)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LumiЯ Chatbot API...")
    create_chat_history_table(DB_CONFIG=DB_CONFIG)
    logger.info("MySQL chat_history table ready.")
    yield
    logger.info("Shutting down LumiЯ Chatbot API...")


app = FastAPI(
    title="LumiЯ Chatbot",
    version="1.0.0",
    lifespan=lifespan
)


# -------------------------
# Health check
# -------------------------
@app.get("/", tags=["Health"])
async def get_status():
    return {"status": "running"}


# -------------------------
# Chat endpoint
# -------------------------
@app.post(
    "/chatbot-rag-agent",
    response_model=ChatResponse,
    tags=["Chatbot"]
)
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    logger.info(f"[USER:{user_id}] {user_message}")

    try:
        # Init chat history
        chat_history = MySQLChatMessageHistory(
            session_id=user_id,
            DB_CONFIG=DB_CONFIG
        )

        # Save user message
        chat_history.add_message(
            message_type="user",
            content=user_message
        )

        # Invoke agent (LangGraph memory via thread_id)
        result = chatbot_agent_executor.invoke({
            "input": user_message,
            "thread_id": user_id
        })

        # Validate agent output
        if not isinstance(result, dict) or "output" not in result:
            raise ValueError("Agent response format invalid")

        bot_response = result["output"]

        # Save AI response
        chat_history.add_message(
            message_type="ai",
            content=bot_response
        )

        logger.info(f"[BOT:{user_id}] {bot_response}")

        return ChatResponse(
            user_id=user_id,
            message=user_message,
            response=bot_response,
        )

    except Exception as e:
        logger.exception("❌ Chatbot error")
        raise HTTPException(
            status_code=500,
            detail=f"Chatbot processing failed: {str(e)}"
        )


# -------------------------
# Local run
# -------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info"
    )
