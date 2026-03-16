from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


class UserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserUpdateRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None


class ApiKeyCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
