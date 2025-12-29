# routers/upload.py

import faiss
from fastapi import APIRouter, Request, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from database import get_db
from dependencies import get_cookie_user
from utils import extract_text, update_vector_store, get_legal_response
import uuid
import pickle  # if needed for vector store reset
router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, db = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("upload.html", {"request": request, "user": user})

@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), db = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")
    
    ext = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".docx", ".doc", ".txt"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, DOC, TXT allowed")

    file_id = str(uuid.uuid4())
    file_path = Path("legal_docs") / f"{file_id}{ext}"

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    text = extract_text(file, str(file_path))
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from file")

    update_vector_store(file_id, text, file.filename)

    # Optional: generate analysis of uploaded document
    analysis_query = f"Provide a brief legal summary of the uploaded document: {text[:500]}..."
    response_data = get_legal_response(analysis_query)

    file_url = f"/legal_docs/{file_id}{ext}"
    return JSONResponse({
        "success": True,
        "filename": file.filename,
        "fileUrl": file_url,
        "answer": response_data["response"],
        "sources": response_data["sources"]
    })

@router.post("/clear-uploads")
async def clear_uploads(request: Request, db = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        for p in Path("legal_docs").glob("*"):
            p.unlink()
        
        global vectorstore, retriever
        from utils import vectorstore, retriever, INDEX_PATH, EMBEDDINGS_PATH
        if vectorstore:
            vectorstore = None
            retriever = None
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        vectorstore = faiss.from_texts([], embeddings)
        vectorstore.save_local(INDEX_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"success": True}