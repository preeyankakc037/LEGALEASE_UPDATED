# routers/chat.py
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import get_db
from models import ChatMessage
from dependencies import get_cookie_user
from utils import get_legal_response
from datetime import datetime
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse


from fastapi import APIRouter, Request, Depends, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates




router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
    
    return templates.TemplateResponse("chatbot.html", {
        "request": request, 
        "user": user, 
        "chat_history": chat_history,
        "now": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    })

@router.post("/query")
async def query(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"}
        )
    
    try:
        data = await request.json()
        user_message = data.get("query", "").strip()
        
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "Query is required"}
            )
        
        print(f"Processing query from {user.username}: {user_message}")
        
        # Save user message
        user_chat_message = ChatMessage(user_id=user.id, message=user_message, is_user=True)
        db.add(user_chat_message)
        db.commit()
        
        # Get response from Gemini + RAG
        response_data = get_legal_response(user_message)
        
        # Save bot response
        bot_chat_message = ChatMessage(user_id=user.id, message=response_data["response"], is_user=False)
        db.add(bot_chat_message)
        db.commit()
        
        is_conversation = len(response_data["sources"]) == 0
        
        return {
            "response": response_data["response"],
            "sources": response_data["sources"],
            "is_conversation": is_conversation
        }
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/chat-history")
async def get_chat_history(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"}
        )
    
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
    return [{"id": msg.id, "message": msg.message, "is_user": msg.is_user, "timestamp": msg.timestamp.isoformat()} for msg in chat_history]

@router.post("/clear-history")
async def clear_chat_history(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication required"}
        )
    
    try:
        count_before = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).count()
        db.query(ChatMessage).filter(ChatMessage.user_id == user.id).delete()
        db.commit()
        return {"message": "Chat history cleared successfully", "deleted_count": count_before}
    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/clear-history-form")
async def clear_chat_history_form(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    try:
        db.query(ChatMessage).filter(ChatMessage.user_id == user.id).delete()
        db.commit()
        return RedirectResponse(url="/chatbot", status_code=303)
    except Exception as e:
        db.rollback()
        return templates.TemplateResponse("error.html", {"request": request, "error_message": str(e)})