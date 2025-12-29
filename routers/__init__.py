# routers/__init__.py
# This file makes the folder a Python package


from .auth import router as auth_router
from .chat import router as chat_router
from .profile import router as profile_router
from .upload import router as upload_router

__all__ = ["auth_router", "chat_router", "profile_router", "upload_router"]