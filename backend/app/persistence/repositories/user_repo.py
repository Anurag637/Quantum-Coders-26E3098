from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    id: str
    username: str
    is_active: bool

class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
        
    async def get_by_id(self, user_id: str) -> Optional[User]:
        # Mock implementation, actual DB queries go here
        return User(id=user_id, username="admin_user", is_active=True)
