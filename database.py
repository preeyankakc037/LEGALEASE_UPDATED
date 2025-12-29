# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base  # Correct import for SQLAlchemy 2.0+

SQLALCHEMY_DATABASE_URL = "sqlite:///./legal_chatbot.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()  # This will now be defined correctly

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()