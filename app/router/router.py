import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.responses import StreamingResponse
from fastapi import APIRouter
from model.chat_model import get_conversation_chain_openai,get_conversation_chain_llama,get_conversation_chain_falcon,get_conversation_chain_bloke,get_conversation_chain_flan, get_vectorstore
from starlette.responses import JSONResponse


router = APIRouter()

from pydantic import BaseModel

chat_responses = {}

class Question(BaseModel):
    question: str

@router.post("/ask")
async def async_ask(question_body: Question, session_id: str = "default"):
    question_text = question_body.question

    vectorstore = get_vectorstore()
    conversation_chain = get_conversation_chain_openai(vectorstore)
    # conversation_chain = get_conversation_chain_llama(vectorstore)
    # conversation_chain = get_conversation_chain_flan(vectorstore)
    # conversation_chain = get_conversation_chain_bloke(vectorstore)
    # conversation_chain = get_conversation_chain_falcon(vectorstore)
    
    
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