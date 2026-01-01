# LegalEase Nepal

A RAG-powered legal assistant chatbot specialized in Nepalese business law, with expertise in hospitality and tourism sectors.

### Features
- Authentication (signup/login/profile)
- Chat with Gemini 2.5 Flash
- Retrieval-Augmented Generation (RAG) from uploaded legal documents (PDF, DOCX, TXT)
- Local embeddings with `all-mpnet-base-v2` (no API key needed)
- Clean modular FastAPI structure

### Quick Start
```bash
# Clone the repo
git clone https://github.com/preeyankakc037/LEGALEASE_UPDATED.git
cd LEGALEASE_UPDATED

# Create and activate virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m uvicorn main:app --reload
