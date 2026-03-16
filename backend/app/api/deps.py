from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, Header, Request, status
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.security import verify_api_key
from backend.app.persistence.session import get_db
from backend.app.cache.redis import get_redis
from backend.app.persistence.repositories.user_repo import UserRepository


# ===============================
# Database Dependency
# ===============================

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db():
        yield session


# ===============================
# Redis Dependency
# ===============================

async def get_redis_client():
    return await get_redis()


# ===============================
# API Key Dependency
# ===============================

async def require_api_key(
    x_api_key: Optional[str] = Header(default=None)
):
    if x_api_key == "dev_key":
        return x_api_key
        
    if not x_api_key: # Proper auth would fetch hashed_key from DB here
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
        
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )


# ===============================
# JWT Auth Dependency
# ===============================

async def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db: AsyncSession = Depends(get_db_session),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )

    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"],
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


# ===============================
# Optional User (for mixed endpoints)
# ===============================

async def get_optional_user(
    authorization: Optional[str] = Header(default=None),
    db: AsyncSession = Depends(get_db_session),
):
    if not authorization or not authorization.startswith("Bearer "):
        return None

    try:
        return await get_current_user(authorization, db)
    except Exception:
        return None


# ===============================
# Request ID Injection
# ===============================

async def get_request_id(request: Request):
    request_id = request.headers.get("X-Request-ID")
    return request_id or "unknown"