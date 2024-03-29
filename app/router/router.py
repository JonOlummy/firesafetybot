import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from fastapi import APIRouter
from app.model.chat_model import get_conversation_chain_openai, get_conversation_chain_ChatCohere, get_vectorstore
from starlette.responses import JSONResponse


sys.path.append(str(Path(__file__).resolve().parent.parent))

router = APIRouter()

from pydantic import BaseModel

chat_responses = {}

class Question(BaseModel):
    question: str

@router.post("/ask")
async def async_ask(question_body: Question, session_id: str = "default"):
    question_text = question_body.question

    vectorstore = await get_vectorstore()
    
    conversation_chain = get_conversation_chain_openai(vectorstore)
    # conversation_chain = get_conversation_chain_ChatCohere(vectorstore)
    

    response = conversation_chain({'question': question_text})
    chat_history = response['chat_history']
    
    human_messages = []
    ai_messages = []

    for i, message in enumerate(chat_history):

        if message.type == "human":
            human_messages.append(message.content)
        else:
            ai_messages.append(message.content)
            

    return JSONResponse(content={"human_messages": human_messages, "ai_messages": ai_messages})
    # chat_responses[session_id] = {"human_messages": human_messages, "ai_messages": ai_messages}
    # return JSONResponse(content=chat_responses[session_id])
    # return chat_history

@router.get("/chat/history")
async def get_chat_history(session_id: str = "default"):
    if session_id in chat_responses:
        return chat_responses[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")