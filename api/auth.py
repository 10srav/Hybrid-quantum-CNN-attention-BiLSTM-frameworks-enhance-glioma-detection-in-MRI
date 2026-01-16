"""
JWT Authentication for the API.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config
from .schemas import TokenData

config = get_config()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# In-memory user store (replace with database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "is_active": True
    },
    "demo": {
        "username": "demo",
        "email": "demo@example.com",
        "hashed_password": pwd_context.hash("demo123"),
        "is_active": True
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[dict]:
    """Get user from database."""
    if username in fake_users_db:
        return fake_users_db[username]
    return None


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Token payload data.
        expires_delta: Token expiration time.
        
    Returns:
        Encoded JWT token.
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=config.api.access_token_expire_minutes
        )
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        config.api.secret_key,
        algorithm=config.api.algorithm
    )
    
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """
    Get the current user from JWT token.
    
    Returns None if no token provided (for optional auth).
    Raises HTTPException if token is invalid.
    """
    if token is None:
        return None
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            config.api.secret_key,
            algorithms=[config.api.algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Get current active user (required auth)."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not current_user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user


def create_user(username: str, email: str, password: str) -> dict:
    """Create a new user."""
    if username in fake_users_db:
        raise ValueError("Username already exists")
    
    user = {
        "username": username,
        "email": email,
        "hashed_password": get_password_hash(password),
        "is_active": True
    }
    
    fake_users_db[username] = user
    
    return user
