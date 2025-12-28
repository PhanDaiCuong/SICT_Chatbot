from fastapi import FastAPI, HTTPException
import os
from agents.db.chat_history import (
    MySQLChatMessageHistory,
    create_chat_history_table,
    get_chat_history_vector_store,
)
from models.chatbot_rag_query import ChatRequest, ChatResponse
from agents.chatbot_rag_agents import chatbot_agent_executor, DB_CONFIG
import uvicorn
 
app = FastAPI(title="SICT Chatbot")

# Initialize Qdrant chat history vectors
chat_history_vectors = get_chat_history_vector_store()

create_chat_history_table(DB_CONFIG=DB_CONFIG)

@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/chatbot-rag-agent", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
        API endpoint to handle user queries and return responses from the chatbot agent.

        Request JSON format:
        {
            "user_id": "<user_id>",
            "message": "<user_input>"
        }

        Response JSON format:
        {
            "user_id": "<user_id>",
            "message": "<user_input>",
            "response": "<bot_response>",
            "error": "<error_message>"
        }
    """
    try: 
        user_id = request.user_id
        user_message = request.message

        # Fetch only relevant chat history (semantic, filtered by session)
        chat_history = MySQLChatMessageHistory(session_id=user_id, DB_CONFIG=DB_CONFIG)
        top_k = int(os.getenv("CHAT_HISTORY_K", "6"))
        history = chat_history_vectors.search_relevant(session_id=user_id, query=user_message, k=top_k)
        
        #Generate response using the chatbot agent
        response = chatbot_agent_executor.invoke({
            "input": user_message,
            "chat_history": history,
        })

        # Save chat history (MySQL + Qdrant vectors)
        chat_history.add_message(message_type="user", content=user_message)
        chat_history_vectors.add_message(session_id=user_id, message_type="user", content=user_message)
        chat_history.add_message(message_type="ai", content=response['output'])
        chat_history_vectors.add_message(session_id=user_id, message_type="ai", content=response['output'])

        return ChatResponse(
                user_id=user_id,
                message=user_message,
                response=response['output'],
            )

    except Exception as e:
        return ChatResponse(
            user_id=request.user_id,
            message=request.message,
            response="",
            error=str(e),
        )
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)