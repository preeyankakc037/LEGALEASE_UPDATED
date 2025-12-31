# utils.py
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Correct modern import
import pickle
from pypdf import PdfReader
from docx import Document
from pathlib import Path
import uuid
from bs4 import BeautifulSoup
import markdown
from models import ChatMessage

# Gemini setup
GEMINI_API_KEY = "your api key "
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"  

# Full SYSTEM_PROMPT
SYSTEM_PROMPT = """
You are a highly knowledgeable and reliable Legal Assistant specialized in Nepalese Business Law, with particular expertise in the Hospitality and Tourism sectors. Your core responsibility is to deliver clear, verified, and step-by-step legal guidance on starting and operating a business in Nepal. Your assistance must cover, but is not limited to:

Company registration and sector-specific licenses
Tax registration, PAN/VAT compliance, and filing
Labor laws, employee contracts, benefits, and workplace compliance
Intellectual Property (IP) protection, including trademarks and copyrights
Tourism and hospitality-specific legal obligations, such as hotel/restaurant licensing, food safety regulations, and tourist activity compliance
Other regulatory acts, such as the Food Act, Tourism Act, Hotel Management Guidelines, and any relevant provincial rules.

Operational Guidelines:
Language Policy:
- If the user writes in Nepali or mixes Nepali and English, respond in formal, professional Nepali.
- If the user writes in English, respond in English.

Accuracy & Sources:
- Rely strictly on verified Nepali laws and procedures.
- Reference official sites like:
  - Office of the Company Registrar – OCR
  - Inland Revenue Department – IRD
  - Department of Industry
  - Department of Tourism
- If any regulation is ambiguous or likely to change, clearly state so and advise consultation with the relevant authority.

Communication Style:
- Use structured, markdown-formatted responses for clarity.
- Never speculate. Never provide outdated or unofficial advice.
- Provide customized responses depending on the business type if specified by the user (e.g., hotel, restaurant, travel agency, trekking company, etc.).
- Add one subtle emoji 🙂 only if the context requires warmth; otherwise maintain a formal tone.

Introduction Rules:
- If asked “Who are you?” or “Tell me about yourself,” give a brief, professional introduction.
- Do not answer the user's main legal query in your introduction.

Scope Clarification:
- You must answer questions related to any Nepalese business law, including laws such as the Food Act, Consumer Protection Act, Tourism Regulations, etc.—even if they are not directly about business setup.

If the user asks a question in Nepali, reply in Nepali. If the user asks in English, reply in English
in questions like tell me about yourself don't give sources 
"""

INDEX_PATH = "faiss_index"
EMBEDDINGS_PATH = "embeddings.pkl"

vectorstore = None
retriever = None

# Initialize embeddings with a reliable local model
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Load or create embeddings and vector store
try:
    print("Loading embeddings...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings = pickle.load(f)
    
    # Safety check - recreate if wrong type
    if not isinstance(embeddings, HuggingFaceEmbeddings):
        raise ValueError("Incompatible embeddings type")
        
    print("Loading FAISS index...")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Vector store loaded successfully")
except Exception as e:
    print(f"Error loading vector store: {e}")
    print("Will create fresh embeddings on first upload")
    embeddings = get_embeddings()
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    vectorstore = None
    retriever = None

def extract_text(file, file_path: str) -> str:
    ext = Path(file.filename).suffix.lower()
    text = ""
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    elif ext in [".docx", ".doc"]:
        doc = Document(file_path)
        text = " ".join(p.text for p in doc.paragraphs)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
    print(f"Extracted text length from {file.filename}: {len(text)}")
    return text.strip()

def update_vector_store(file_id: str, text: str, filename: str):
    global vectorstore, retriever
    if not text.strip():
        return
    
    embeddings = get_embeddings()
    
    if vectorstore is None:
        vectorstore = FAISS.from_texts(
            [text],
            embeddings,
            metadatas=[{"source": filename, "file_id": file_id}],
            ids=[file_id]
        )
    else:
        vectorstore.add_texts(
            [text],
            metadatas=[{"source": filename, "file_id": file_id}],
            ids=[file_id]
        )
    
    vectorstore.save_local(INDEX_PATH)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Vector store updated successfully")

def get_legal_response(query: str, chat_history=None) -> dict:
    sources = []
    context = ""
    found_in_pdf = False
    
    if retriever:
        try:
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                context += f"\n\nDocument: {source}\n{doc.page_content}"
                if source.lower().endswith('.pdf'):
                    found_in_pdf = True
                    sources.append({
                        "content": doc.page_content[:200] + "...",
                        "source": source
                    })
            print(f"Retrieved {len(docs)} relevant documents for RAG")
        except Exception as e:
            print(f"RAG Error: {e}")

    prompt_text = f"""
{SYSTEM_PROMPT}

RELEVANT DOCUMENT EXCERPTS:
{context}

USER QUERY: {query}

Provide a clear, accurate, and professional response in markdown format. Use the document excerpts when relevant.
"""

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=GenerationConfig(
                temperature=0.2,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
            )
        )
        response = model.generate_content(prompt_text)
        raw_answer = response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        print(f"Gemini Error: {str(e)}")
        raw_answer = "Sorry, the AI service is temporarily unavailable."

    raw_answer = raw_answer.replace("```html", "").replace("```", "")
    html_answer = markdown.markdown(raw_answer)
    soup = BeautifulSoup(html_answer, 'html.parser')
    
    for tag in soup.find_all(['h1','h2','h3','h4','h5','h6']):
        tag['class'] = tag.get('class', []) + ['legal-heading']
    for tag in soup.find_all('p'):
        tag['class'] = tag.get('class', []) + ['legal-paragraph']
    for tag in soup.find_all(['ul','ol']):
        tag['class'] = tag.get('class', []) + ['legal-list']
    for tag in soup.find_all('li'):
        tag['class'] = tag.get('class', []) + ['legal-list-item']
    for tag in soup.find_all(['strong','b']):
        tag['class'] = tag.get('class', []) + ['legal-bold']
    for tag in soup.find_all(['em','i']):
        tag['class'] = tag.get('class', []) + ['legal-italic']

    formatted_answer = str(soup)
    show_sources = found_in_pdf and len(query.split()) > 2

    return {
        "response": formatted_answer,
        "sources": sources if show_sources else []
    }
