# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request, Depends

from database import engine
from models import Base
from routers import auth, chat, profile, upload
from dependencies import get_cookie_user
from database import get_db

app = FastAPI(title="Legal Chatbot")

Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(profile.router)
app.include_router(upload.router)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db = Depends(get_db)):
    user = get_cookie_user(request, db)
    return templates.TemplateResponse("home.html", {"request": request, "user": user})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, db = Depends(get_db)):
    user = get_cookie_user(request, db)
    return templates.TemplateResponse("about.html", {"request": request, "user": user})

@app.get("/connect", response_class=HTMLResponse)
async def connect(request: Request, db = Depends(get_db)):
    user = get_cookie_user(request, db)
    return templates.TemplateResponse("connect.html", {"request": request, "user": user})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)