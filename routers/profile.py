# routers/profile.py
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import get_db
from models import ChatMessage
from dependencies import get_cookie_user


router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/profile", response_class=HTMLResponse)
async def profile(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
    return templates.TemplateResponse("profile.html", {"request": request, "user": user, "chat_history": chat_history})

@router.post("/update-profile", response_class=HTMLResponse)
async def update_profile(
    request: Request,
    full_name: str = Form(None),
    email: str = Form(None),
    phone: str = Form(None),
    db: Session = Depends(get_db)
):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    if full_name:
        user.full_name = full_name
    if email:
        user.email = email
    if phone:
        user.phone = phone
    
    db.commit()
    db.refresh(user)
    
    return RedirectResponse(url="/profile", status_code=303)