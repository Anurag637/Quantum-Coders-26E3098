from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from backend.app.api.deps import (
    get_db_session,
    require_api_key,
    get_current_user,
)
from backend.app.schemas.user import (
    UserResponse,
    UserUpdateRequest,
    ApiKeyCreateRequest,
    ApiKeyResponse,
)
from backend.app.services.user_service import UserService


router = APIRouter()


# ===============================
# Get Current User Profile
# ===============================

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
    user=Depends(get_current_user),
):
    service = UserService(db)
    return await service.get_user(user.id)


# ===============================
# Update Profile
# ===============================

@router.put("/me", response_model=UserResponse)
async def update_profile(
    payload: UserUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
    user=Depends(get_current_user),
):
    service = UserService(db)
    return await service.update_user(user.id, payload)


# ===============================
# Create API Key
# ===============================

@router.post("/me/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    payload: ApiKeyCreateRequest,
    db: AsyncSession = Depends(get_db_session),
    user=Depends(get_current_user),
):
    service = UserService(db)
    return await service.create_api_key(user.id, payload)


# ===============================
# List API Keys
# ===============================

@router.get("/me/api-keys", response_model=List[ApiKeyResponse])
async def list_api_keys(
    db: AsyncSession = Depends(get_db_session),
    user=Depends(get_current_user),
):
    service = UserService(db)
    return await service.list_api_keys(user.id)


# ===============================
# Delete API Key
# ===============================

@router.delete("/me/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_db_session),
    user=Depends(get_current_user),
):
    service = UserService(db)
    await service.delete_api_key(user.id, key_id)
    return None


# ===============================
# Admin: List All Users
# ===============================

@router.get("/admin", response_model=List[UserResponse])
async def list_users(
    db: AsyncSession = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
):
    """
    Admin-only endpoint to list all users.
    """
    service = UserService(db)
    return await service.list_users()