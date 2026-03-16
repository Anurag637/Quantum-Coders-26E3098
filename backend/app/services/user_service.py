import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from backend.app.schemas.user import (
    UserResponse,
    UserUpdateRequest,
    ApiKeyCreateRequest,
    ApiKeyResponse,
)


# In-memory store for users and API keys (no separate DB table needed for this demo)
_USERS: dict[str, dict] = {}
_API_KEYS: dict[str, list] = {}


def _get_or_create_user(user_id: str) -> dict:
    if user_id not in _USERS:
        _USERS[user_id] = {
            "id": user_id,
            "username": f"user_{user_id[:8]}",
            "email": None,
            "is_active": True,
            "created_at": None,
        }
    return _USERS[user_id]


class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user(self, user_id: str) -> UserResponse:
        data = _get_or_create_user(user_id)
        return UserResponse(**data)

    async def update_user(self, user_id: str, payload: UserUpdateRequest) -> UserResponse:
        data = _get_or_create_user(user_id)
        if payload.username is not None:
            data["username"] = payload.username
        if payload.email is not None:
            data["email"] = payload.email
        return UserResponse(**data)

    async def create_api_key(self, user_id: str, payload: ApiKeyCreateRequest) -> ApiKeyResponse:
        key_id = str(uuid.uuid4())
        raw_key = f"sk-{uuid.uuid4().hex}"
        key_entry = {
            "id": key_id,
            "name": payload.name,
            "key_prefix": raw_key[:8] + "...",
            "description": payload.description,
            "created_at": None,
        }
        _API_KEYS.setdefault(user_id, []).append(key_entry)
        return ApiKeyResponse(**key_entry)

    async def list_api_keys(self, user_id: str) -> list[ApiKeyResponse]:
        return [ApiKeyResponse(**k) for k in _API_KEYS.get(user_id, [])]

    async def delete_api_key(self, user_id: str, key_id: str) -> None:
        keys = _API_KEYS.get(user_id, [])
        _API_KEYS[user_id] = [k for k in keys if k["id"] != key_id]

    async def list_users(self) -> list[UserResponse]:
        return [UserResponse(**u) for u in _USERS.values()]
