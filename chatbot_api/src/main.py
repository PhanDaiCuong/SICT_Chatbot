from fastapi import FastAPI, HTTPException
from agents.db.chat_history import MySQLChatMessageHistory, create_chat_history_table
from models.chatbot_rag_query import ChatRequest, ChatResponse
from agents.chatbot_rag_agents import chatbot_agent_executor, DB_CONFIG
import uvicorn
 
app = FastAPI(title="Vietnamese History and Culture Chatbot")

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

        # Fetch chat history for the user
        chat_history = MySQLChatMessageHistory(session_id=user_id, DB_CONFIG=DB_CONFIG)
        history = chat_history.load_messages() 
        
        #Generate response using the chatbot agent
        response = chatbot_agent_executor.invoke({
            "input": user_message,
            "chat_history": history,
        })

        # Save chat history
        chat_history.add_message(message_type="user", content=user_message)
        chat_history.add_message(message_type="ai", content=response['output'])

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