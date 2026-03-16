"""
Admin Endpoints - Production Ready
System administration, configuration management, user management, and maintenance operations
Secure endpoints requiring admin privileges with audit logging and rate limiting
"""

from fastapi import APIRouter, Request, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Dict, Any, List, Optional, Union
import time
import uuid
import asyncio
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field, validator

from core.logging import get_logger
from core.security import verify_admin, get_current_user, hash_password, verify_password
from core.rate_limiter import rate_limit
from core.exceptions import AdminPermissionError, ConfigurationError
from config import settings
from database.session import get_db_session, check_db_connection
from database.repositories.user_repository import UserRepository
from database.repositories.model_repository import ModelRepository
from database.repositories.metrics_repository import MetricsRepository
from database.repositories.request_log_repository import RequestLogRepository
from models.model_manager import ModelManager
from models.model_registry import ModelRegistry
from cache.cache_manager import CacheManager
from cache.semantic_cache import SemanticCache
from monitoring.metrics import MetricsCollector
from monitoring.alerting import AlertManager
from monitoring.health_check import HealthChecker
from services.user_service import UserService
from utils.config_manager import ConfigManager
from utils.backup_manager import BackupManager

# Initialize router
router = APIRouter(prefix="/admin", tags=["Admin"])

# Initialize logger
logger = get_logger(__name__)

# Initialize services
user_repository = UserRepository()
user_service = UserService()
model_manager = ModelManager()
model_registry = ModelRegistry()
cache_manager = CacheManager()
semantic_cache = SemanticCache()
metrics_collector = MetricsCollector()
alert_manager = AlertManager()
health_checker = HealthChecker()
config_manager = ConfigManager()
backup_manager = BackupManager()
request_log_repository = RequestLogRepository()

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class AdminActionResponse(BaseModel):
    """Base response schema for admin actions"""
    status: str = Field(..., description="Action status")
    message: str = Field(..., description="Human-readable message")
    timestamp: datetime = Field(..., description="Action timestamp")
    admin_id: str = Field(..., description="Admin user ID")
    request_id: str = Field(..., description="Request ID for tracing")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class UserCreateRequest(BaseModel):
    """User creation request schema"""
    email: str = Field(..., description="User email")
    username: str = Field(..., description="Username", min_length=3, max_length=50)
    password: str = Field(..., description="Password", min_length=8)
    is_admin: bool = Field(False, description="Whether user has admin privileges")
    rate_limit_quota: int = Field(100, description="Rate limit quota per minute", ge=1, le=10000)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional user metadata")
    
    @validator('email')
    def validate_email(cls, v):
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v


class UserUpdateRequest(BaseModel):
    """User update request schema"""
    email: Optional[str] = Field(None, description="Updated email")
    username: Optional[str] = Field(None, description="Updated username", min_length=3, max_length=50)
    password: Optional[str] = Field(None, description="Updated password", min_length=8)
    is_active: Optional[bool] = Field(None, description="Whether user is active")
    is_admin: Optional[bool] = Field(None, description="Whether user has admin privileges")
    rate_limit_quota: Optional[int] = Field(None, description="Rate limit quota per minute", ge=1, le=10000)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional user metadata")


class UserResponse(BaseModel):
    """User response schema"""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    username: str = Field(..., description="Username")
    is_active: bool = Field(..., description="Whether user is active")
    is_admin: bool = Field(..., description="Whether user has admin privileges")
    api_key: Optional[str] = Field(None, description="API key (masked)")
    rate_limit_quota: int = Field(..., description="Rate limit quota")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    total_requests: int = Field(0, description="Total API requests")
    metadata: Optional[Dict[str, Any]] = Field(None, description="User metadata")


class SystemConfigResponse(BaseModel):
    """System configuration response schema"""
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment")
    debug: bool = Field(..., description="Debug mode")
    features: Dict[str, bool] = Field(..., description="Enabled features")
    limits: Dict[str, Any] = Field(..., description="System limits")
    cache: Dict[str, Any] = Field(..., description="Cache configuration")
    rate_limits: Dict[str, Any] = Field(..., description="Rate limit configuration")
    models: Dict[str, Any] = Field(..., description="Model configuration")
    monitoring: Dict[str, Any] = Field(..., description="Monitoring configuration")
    updated_at: datetime = Field(..., description="Configuration last updated")


class SystemConfigUpdateRequest(BaseModel):
    """System configuration update request"""
    debug: Optional[bool] = Field(None, description="Debug mode")
    features: Optional[Dict[str, bool]] = Field(None, description="Enable/disable features")
    limits: Optional[Dict[str, Any]] = Field(None, description="Update system limits")
    cache_ttl: Optional[int] = Field(None, description="Cache TTL in seconds", ge=60, le=86400)
    semantic_threshold: Optional[float] = Field(None, description="Semantic similarity threshold", ge=0.5, le=1.0)
    rate_limit_default: Optional[int] = Field(None, description="Default rate limit", ge=1, le=10000)


class CacheStatsResponse(BaseModel):
    """Cache statistics response"""
    redis: Dict[str, Any] = Field(..., description="Redis cache stats")
    semantic: Dict[str, Any] = Field(..., description="Semantic cache stats")
    routing: Dict[str, Any] = Field(..., description="Routing decision cache stats")
    total_size_mb: float = Field(..., description="Total cache size in MB")
    total_entries: int = Field(..., description="Total cache entries")
    hit_rate_avg: float = Field(..., description="Average hit rate")
    timestamp: datetime = Field(..., description="Stats timestamp")


class SystemHealthResponse(BaseModel):
    """System health response"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Service health")
    resources: Dict[str, Any] = Field(..., description="Resource utilization")
    uptime_seconds: float = Field(..., description="System uptime")
    version: str = Field(..., description="Application version")
    alerts: List[Dict[str, Any]] = Field(..., description="Active alerts")


class BackupResponse(BaseModel):
    """Backup response schema"""
    backup_id: str = Field(..., description="Backup ID")
    filename: str = Field(..., description="Backup filename")
    size_bytes: int = Field(..., description="Backup size in bytes")
    created_at: datetime = Field(..., description="Creation timestamp")
    status: str = Field(..., description="Backup status")
    components: List[str] = Field(..., description="Backed up components")


class AuditLogEntry(BaseModel):
    """Audit log entry schema"""
    id: str = Field(..., description="Log entry ID")
    timestamp: datetime = Field(..., description="Entry timestamp")
    admin_id: str = Field(..., description="Admin user ID")
    admin_username: str = Field(..., description="Admin username")
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Resource type (user, model, config, etc.)")
    resource_id: Optional[str] = Field(None, description="Resource ID")
    changes: Optional[Dict[str, Any]] = Field(None, description="Changes made")
    ip_address: str = Field(..., description="Admin IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    status: str = Field(..., description="Action status")
    error: Optional[str] = Field(None, description="Error message if failed")
    request_id: str = Field(..., description="Request ID for tracing")


# ============================================================================
# ADMIN AUTHENTICATION & AUTHORIZATION
# ============================================================================

async def verify_admin_access(request: Request) -> Dict[str, Any]:
    """
    Verify admin access and return admin user info.
    Used as dependency for all admin endpoints.
    """
    api_key = request.headers.get("X-API-Key")
    auth_header = request.headers.get("Authorization")
    
    if not api_key and not auth_header:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "AuthenticationRequired",
                "message": "API key or JWT token required"
            }
        )
    
    # Verify admin access
    from core.security import verify_admin
    admin = await verify_admin(api_key or auth_header)
    
    if not admin:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "AdminRequired",
                "message": "Admin privileges required for this endpoint"
            }
        )
    
    return admin


async def log_admin_action(
    request: Request,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    changes: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error: Optional[str] = None
):
    """
    Log admin actions for audit trail.
    """
    admin = getattr(request.state, "admin_user", None)
    
    log_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "admin_id": admin.get("id") if admin else "unknown",
        "admin_username": admin.get("username") if admin else "unknown",
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "changes": changes,
        "ip_address": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent"),
        "status": status,
        "error": error,
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }
    
    # Store in database
    from database.repositories.audit_repository import AuditRepository
    audit_repo = AuditRepository()
    await audit_repo.log_action(log_entry)
    
    # Also log to file
    logger.info(
        "admin_action",
        **log_entry
    )


# ============================================================================
# USER MANAGEMENT ENDPOINTS
# ============================================================================

@router.get(
    "/users",
    summary="List Users",
    description="""
    List all users in the system.
    
    Returns paginated list of users with their details, usage statistics, and status.
    Supports filtering by active status, admin role, and search.
    
    Admin only endpoint.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def list_users(
    request: Request,
    active_only: bool = Query(False, description="Show only active users"),
    admin_only: bool = Query(False, description="Show only admin users"),
    search: Optional[str] = Query(None, description="Search by username or email"),
    limit: int = Query(50, description="Number of users to return", ge=1, le=100),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """
    List all users with filtering and pagination.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "list_users_requested",
        request_id=request_id,
        admin_id=admin.get("id")
    )
    
    try:
        # Get users from repository
        users = await user_repository.get_users(
            active_only=active_only,
            admin_only=admin_only,
            search=search,
            limit=limit,
            offset=offset
        )
        
        # Get total count
        total = await user_repository.get_user_count(
            active_only=active_only,
            admin_only=admin_only,
            search=search
        )
        
        # Get usage statistics for each user
        user_responses = []
        for user in users:
            usage = await request_log_repository.get_user_usage(
                user_id=user["id"],
                start_time=datetime.now() - timedelta(days=30),
                end_time=datetime.now()
            )
            
            # Mask API key for security
            api_key = user.get("api_key")
            if api_key:
                api_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            
            user_responses.append(UserResponse(
                id=user["id"],
                email=user["email"],
                username=user["username"],
                is_active=user["is_active"],
                is_admin=user["is_admin"],
                api_key=api_key,
                rate_limit_quota=user.get("rate_limit_quota", 100),
                created_at=user["created_at"],
                last_login=user.get("last_login"),
                total_requests=usage.get("total_requests", 0),
                metadata=user.get("metadata")
            ))
        
        response = {
            "total": total,
            "limit": limit,
            "offset": offset,
            "users": [user.dict() for user in user_responses],
            "request_id": request_id,
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        await log_admin_action(
            request=request,
            action="list_users",
            resource_type="user",
            status="success"
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "list_users_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="list_users",
            resource_type="user",
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to list users",
                "request_id": request_id
            }
        )


@router.post(
    "/users",
    summary="Create User",
    description="""
    Create a new user account.
    
    Creates user with specified permissions and rate limits.
    Generates API key automatically.
    
    Admin only endpoint.
    """,
    response_model=UserResponse,
    status_code=201,
    dependencies=[Depends(rate_limit(limit=20, period=60))]
)
async def create_user(
    request: Request,
    user_data: UserCreateRequest,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> UserResponse:
    """
    Create a new user account.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
    logger.info(
        "create_user_requested",
        request_id=request_id,
        admin_id=admin.get("id"),
        username=user_data.username,
        email=user_data.email
    )
    
    try:
        # Check if user already exists
        existing = await user_repository.get_user_by_email(user_data.email)
        if existing:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "UserExists",
                    "message": f"User with email '{user_data.email}' already exists",
                    "request_id": request_id
                }
            )
        
        existing = await user_repository.get_user_by_username(user_data.username)
        if existing:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "UserExists",
                    "message": f"Username '{user_data.username}' already taken",
                    "request_id": request_id
                }
            )
        
        # Hash password
        hashed_password = hash_password(user_data.password)
        
        # Generate API key
        import secrets
        api_key = f"llm_{secrets.token_urlsafe(32)}"
        
        # Create user
        user = await user_repository.create_user({
            "email": user_data.email,
            "username": user_data.username,
            "hashed_password": hashed_password,
            "api_key": api_key,
            "is_admin": user_data.is_admin,
            "is_active": True,
            "rate_limit_quota": user_data.rate_limit_quota,
            "metadata": user_data.metadata or {}
        })
        
        # Mask API key for response
        masked_api_key = api_key[:8] + "..." + api_key[-4:]
        
        response = UserResponse(
            id=user["id"],
            email=user["email"],
            username=user["username"],
            is_active=user["is_active"],
            is_admin=user["is_admin"],
            api_key=masked_api_key,
            rate_limit_quota=user["rate_limit_quota"],
            created_at=user["created_at"],
            last_login=user.get("last_login"),
            total_requests=0,
            metadata=user.get("metadata")
        )
        
        await log_admin_action(
            request=request,
            action="create_user",
            resource_type="user",
            resource_id=user["id"],
            changes={
                "username": user_data.username,
                "email": user_data.email,
                "is_admin": user_data.is_admin,
                "rate_limit_quota": user_data.rate_limit_quota
            },
            status="success"
        )
        
        logger.info(
            "create_user_completed",
            request_id=request_id,
            user_id=user["id"],
            response_time_ms=round((time.time() - start_time) * 1000, 2)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "create_user_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="create_user",
            resource_type="user",
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to create user",
                "request_id": request_id
            }
        )


@router.get(
    "/users/{user_id}",
    summary="Get User",
    description="""
    Get detailed information about a specific user.
    
    Returns user profile, permissions, usage statistics, and API key.
    
    Admin only endpoint.
    """,
    response_model=UserResponse,
    dependencies=[Depends(rate_limit(limit=60, period=60))]
)
async def get_user(
    request: Request,
    user_id: str,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> UserResponse:
    """
    Get detailed information about a specific user.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        user = await user_repository.get_user(user_id)
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "UserNotFound",
                    "message": f"User '{user_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Get usage statistics
        usage = await request_log_repository.get_user_usage(
            user_id=user_id,
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now()
        )
        
        # Mask API key
        api_key = user.get("api_key")
        if api_key:
            api_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        
        response = UserResponse(
            id=user["id"],
            email=user["email"],
            username=user["username"],
            is_active=user["is_active"],
            is_admin=user["is_admin"],
            api_key=api_key,
            rate_limit_quota=user.get("rate_limit_quota", 100),
            created_at=user["created_at"],
            last_login=user.get("last_login"),
            total_requests=usage.get("total_requests", 0),
            metadata=user.get("metadata")
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_user_failed",
            request_id=request_id,
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to get user '{user_id}'",
                "request_id": request_id
            }
        )


@router.put(
    "/users/{user_id}",
    summary="Update User",
    description="""
    Update an existing user account.
    
    Can update email, username, password, permissions, rate limits, and status.
    
    Admin only endpoint.
    """,
    response_model=UserResponse,
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def update_user(
    request: Request,
    user_id: str,
    user_data: UserUpdateRequest,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> UserResponse:
    """
    Update an existing user account.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "update_user_requested",
        request_id=request_id,
        admin_id=admin.get("id"),
        user_id=user_id
    )
    
    try:
        # Check if user exists
        existing = await user_repository.get_user(user_id)
        if not existing:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "UserNotFound",
                    "message": f"User '{user_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Prevent admin from modifying themselves to non-admin
        if user_id == admin.get("id") and user_data.is_admin is False:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "CannotRemoveOwnAdmin",
                    "message": "Cannot remove admin privileges from your own account",
                    "request_id": request_id
                }
            )
        
        # Prepare updates
        updates = {}
        changes = {}
        
        if user_data.email and user_data.email != existing["email"]:
            # Check if email is taken
            email_exists = await user_repository.get_user_by_email(user_data.email)
            if email_exists and email_exists["id"] != user_id:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "EmailExists",
                        "message": f"Email '{user_data.email}' is already in use",
                        "request_id": request_id
                    }
                )
            updates["email"] = user_data.email
            changes["email"] = {"old": existing["email"], "new": user_data.email}
        
        if user_data.username and user_data.username != existing["username"]:
            # Check if username is taken
            username_exists = await user_repository.get_user_by_username(user_data.username)
            if username_exists and username_exists["id"] != user_id:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "UsernameExists",
                        "message": f"Username '{user_data.username}' is already taken",
                        "request_id": request_id
                    }
                )
            updates["username"] = user_data.username
            changes["username"] = {"old": existing["username"], "new": user_data.username}
        
        if user_data.password:
            updates["hashed_password"] = hash_password(user_data.password)
            changes["password_updated"] = True
        
        if user_data.is_active is not None:
            updates["is_active"] = user_data.is_active
            changes["is_active"] = {"old": existing["is_active"], "new": user_data.is_active}
        
        if user_data.is_admin is not None:
            updates["is_admin"] = user_data.is_admin
            changes["is_admin"] = {"old": existing["is_admin"], "new": user_data.is_admin}
        
        if user_data.rate_limit_quota is not None:
            updates["rate_limit_quota"] = user_data.rate_limit_quota
            changes["rate_limit_quota"] = {
                "old": existing["rate_limit_quota"],
                "new": user_data.rate_limit_quota
            }
        
        if user_data.metadata is not None:
            updates["metadata"] = {**existing.get("metadata", {}), **user_data.metadata}
            changes["metadata_updated"] = True
        
        if updates:
            await user_repository.update_user(user_id, updates)
        
        # Get updated user
        updated_user = await user_repository.get_user(user_id)
        
        # Mask API key
        api_key = updated_user.get("api_key")
        if api_key:
            api_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        
        # Get usage statistics
        usage = await request_log_repository.get_user_usage(
            user_id=user_id,
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now()
        )
        
        response = UserResponse(
            id=updated_user["id"],
            email=updated_user["email"],
            username=updated_user["username"],
            is_active=updated_user["is_active"],
            is_admin=updated_user["is_admin"],
            api_key=api_key,
            rate_limit_quota=updated_user.get("rate_limit_quota", 100),
            created_at=updated_user["created_at"],
            last_login=updated_user.get("last_login"),
            total_requests=usage.get("total_requests", 0),
            metadata=updated_user.get("metadata")
        )
        
        await log_admin_action(
            request=request,
            action="update_user",
            resource_type="user",
            resource_id=user_id,
            changes=changes,
            status="success"
        )
        
        logger.info(
            "update_user_completed",
            request_id=request_id,
            user_id=user_id,
            changes=list(changes.keys())
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "update_user_failed",
            request_id=request_id,
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="update_user",
            resource_type="user",
            resource_id=user_id,
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to update user '{user_id}'",
                "request_id": request_id
            }
        )


@router.delete(
    "/users/{user_id}",
    summary="Delete User",
    description="""
    Delete a user account.
    
    Permanently removes user and all associated data.
    Cannot delete your own admin account.
    
    Admin only endpoint.
    """,
    response_model=AdminActionResponse,
    dependencies=[Depends(rate_limit(limit=10, period=60))]
)
async def delete_user(
    request: Request,
    user_id: str,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> AdminActionResponse:
    """
    Delete a user account.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "delete_user_requested",
        request_id=request_id,
        admin_id=admin.get("id"),
        user_id=user_id
    )
    
    try:
        # Check if user exists
        user = await user_repository.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "UserNotFound",
                    "message": f"User '{user_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Prevent deleting own account
        if user_id == admin.get("id"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "CannotDeleteOwnAccount",
                    "message": "Cannot delete your own admin account",
                    "request_id": request_id
                }
            )
        
        # Delete user
        await user_repository.delete_user(user_id)
        
        response = AdminActionResponse(
            status="deleted",
            message=f"User '{user['username']}' deleted successfully",
            timestamp=datetime.now(),
            admin_id=admin.get("id"),
            request_id=request_id,
            details={"user_id": user_id, "username": user["username"]}
        )
        
        await log_admin_action(
            request=request,
            action="delete_user",
            resource_type="user",
            resource_id=user_id,
            changes={"username": user["username"]},
            status="success"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "delete_user_failed",
            request_id=request_id,
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="delete_user",
            resource_type="user",
            resource_id=user_id,
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to delete user '{user_id}'",
                "request_id": request_id
            }
        )


@router.post(
    "/users/{user_id}/reset-api-key",
    summary="Reset User API Key",
    description="""
    Reset a user's API key.
    
    Generates a new API key and invalidates the old one.
    Returns the new API key (only shown once).
    
    Admin only endpoint.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=20, period=60))]
)
async def reset_user_api_key(
    request: Request,
    user_id: str,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """
    Reset a user's API key.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "reset_api_key_requested",
        request_id=request_id,
        admin_id=admin.get("id"),
        user_id=user_id
    )
    
    try:
        # Check if user exists
        user = await user_repository.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "UserNotFound",
                    "message": f"User '{user_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Generate new API key
        import secrets
        new_api_key = f"llm_{secrets.token_urlsafe(32)}"
        
        # Update user
        await user_repository.update_user(user_id, {"api_key": new_api_key})
        
        # Mask for logging
        masked_key = new_api_key[:8] + "..." + new_api_key[-4:]
        
        response = {
            "status": "reset",
            "user_id": user_id,
            "username": user["username"],
            "api_key": new_api_key,  # Full key - only shown once!
            "message": "API key reset successfully. Store this key securely - it won't be shown again.",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        await log_admin_action(
            request=request,
            action="reset_api_key",
            resource_type="user",
            resource_id=user_id,
            changes={"api_key_reset": True},
            status="success"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "reset_api_key_failed",
            request_id=request_id,
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="reset_api_key",
            resource_type="user",
            resource_id=user_id,
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to reset API key for user '{user_id}'",
                "request_id": request_id
            }
        )


# ============================================================================
# SYSTEM CONFIGURATION ENDPOINTS
# ============================================================================

@router.get(
    "/config",
    summary="Get System Configuration",
    description="""
    Get current system configuration.
    
    Returns all configurable system settings, feature flags, and limits.
    
    Admin only endpoint.
    """,
    response_model=SystemConfigResponse,
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_system_config(
    request: Request,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> SystemConfigResponse:
    """
    Get current system configuration.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Load current config
        config = await config_manager.get_config()
        
        response = SystemConfigResponse(
            version=settings.version,
            environment=settings.environment.value,
            debug=settings.debug,
            features=config.get("features", {}),
            limits=config.get("limits", {}),
            cache=config.get("cache", {}),
            rate_limits=config.get("rate_limits", {}),
            models=config.get("models", {}),
            monitoring=config.get("monitoring", {}),
            updated_at=config.get("updated_at", datetime.now())
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "get_system_config_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to get system configuration",
                "request_id": request_id
            }
        )


@router.put(
    "/config",
    summary="Update System Configuration",
    description="""
    Update system configuration.
    
    Modify feature flags, system limits, cache settings, and more.
    Changes take effect immediately without restart.
    
    Admin only endpoint.
    """,
    response_model=SystemConfigResponse,
    dependencies=[Depends(rate_limit(limit=10, period=60))]
)
async def update_system_config(
    request: Request,
    config_updates: SystemConfigUpdateRequest,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> SystemConfigResponse:
    """
    Update system configuration.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "update_config_requested",
        request_id=request_id,
        admin_id=admin.get("id")
    )
    
    try:
        # Get current config
        current_config = await config_manager.get_config()
        
        # Prepare updates
        updates = {}
        changes = {}
        
        if config_updates.debug is not None:
            updates["debug"] = config_updates.debug
            changes["debug"] = {"old": settings.debug, "new": config_updates.debug}
            settings.debug = config_updates.debug
        
        if config_updates.features:
            for feature, enabled in config_updates.features.items():
                if feature in current_config.get("features", {}):
                    updates[f"features.{feature}"] = enabled
                    changes[f"feature_{feature}"] = {
                        "old": current_config["features"].get(feature),
                        "new": enabled
                    }
        
        if config_updates.limits:
            for limit, value in config_updates.limits.items():
                updates[f"limits.{limit}"] = value
                changes[f"limit_{limit}"] = {
                    "old": current_config.get("limits", {}).get(limit),
                    "new": value
                }
        
        if config_updates.cache_ttl:
            updates["cache.ttl"] = config_updates.cache_ttl
            changes["cache_ttl"] = {
                "old": current_config.get("cache", {}).get("ttl"),
                "new": config_updates.cache_ttl
            }
            settings.cache.default_ttl = config_updates.cache_ttl
        
        if config_updates.semantic_threshold:
            updates["cache.semantic_threshold"] = config_updates.semantic_threshold
            changes["semantic_threshold"] = {
                "old": current_config.get("cache", {}).get("semantic_threshold"),
                "new": config_updates.semantic_threshold
            }
            settings.cache.similarity_threshold = config_updates.semantic_threshold
        
        if config_updates.rate_limit_default:
            updates["rate_limits.default"] = config_updates.rate_limit_default
            changes["rate_limit_default"] = {
                "old": current_config.get("rate_limits", {}).get("default"),
                "new": config_updates.rate_limit_default
            }
            settings.rate_limit.default_requests = config_updates.rate_limit_default
        
        # Apply updates
        if updates:
            await config_manager.update_config(updates)
        
        # Get updated config
        updated_config = await config_manager.get_config()
        
        response = SystemConfigResponse(
            version=settings.version,
            environment=settings.environment.value,
            debug=settings.debug,
            features=updated_config.get("features", {}),
            limits=updated_config.get("limits", {}),
            cache=updated_config.get("cache", {}),
            rate_limits=updated_config.get("rate_limits", {}),
            models=updated_config.get("models", {}),
            monitoring=updated_config.get("monitoring", {}),
            updated_at=datetime.now()
        )
        
        await log_admin_action(
            request=request,
            action="update_config",
            resource_type="config",
            changes=changes,
            status="success"
        )
        
        logger.info(
            "update_config_completed",
            request_id=request_id,
            changes=list(changes.keys())
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "update_config_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="update_config",
            resource_type="config",
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to update system configuration",
                "request_id": request_id
            }
        )


@router.post(
    "/config/reload",
    summary="Reload Configuration",
    description="""
    Reload configuration from files.
    
    Reads configuration files and applies changes without restart.
    Useful after manual configuration file edits.
    
    Admin only endpoint.
    """,
    response_model=AdminActionResponse,
    dependencies=[Depends(rate_limit(limit=5, period=60))]
)
async def reload_config(
    request: Request,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> AdminActionResponse:
    """
    Reload configuration from files.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "reload_config_requested",
        request_id=request_id,
        admin_id=admin.get("id")
    )
    
    try:
        # Reload config from files
        await config_manager.reload_from_files()
        
        # Reload model registry
        await model_registry.load_from_config(settings.model.config_path)
        
        # Reload routing rules
        from router.prompt_router import PromptRouter
        router = PromptRouter()
        await router.reload_rules()
        
        response = AdminActionResponse(
            status="reloaded",
            message="Configuration reloaded successfully",
            timestamp=datetime.now(),
            admin_id=admin.get("id"),
            request_id=request_id,
            details={
                "model_config": settings.model.config_path,
                "routing_config": settings.model.routing_config_path,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await log_admin_action(
            request=request,
            action="reload_config",
            resource_type="config",
            status="success"
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "reload_config_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="reload_config",
            resource_type="config",
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to reload configuration",
                "request_id": request_id
            }
        )


# ============================================================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================================================

@router.get(
    "/cache/stats",
    summary="Get Cache Statistics",
    description="""
    Get detailed cache statistics.
    
    Returns hit rates, sizes, entry counts, and performance metrics
    for all cache types (Redis, semantic, routing decisions).
    
    Admin only endpoint.
    """,
    response_model=CacheStatsResponse,
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_cache_stats(
    request: Request,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> CacheStatsResponse:
    """
    Get detailed cache statistics.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Get Redis cache stats
        redis_stats = await cache_manager.get_stats()
        
        # Get semantic cache stats
        semantic_stats = await semantic_cache.get_stats()
        
        # Get routing decision cache stats
        from router.prompt_router import PromptRouter
        router = PromptRouter()
        routing_stats = await router.get_cache_stats()
        
        # Calculate total size
        total_size_mb = (
            redis_stats.get("used_memory", 0) +
            semantic_stats.get("size_bytes", 0) +
            routing_stats.get("size_bytes", 0)
        ) / (1024 * 1024)
        
        # Calculate total entries
        total_entries = (
            redis_stats.get("keys", 0) +
            semantic_stats.get("entries", 0) +
            routing_stats.get("entries", 0)
        )
        
        # Calculate average hit rate
        hit_rates = []
        if redis_stats.get("hits", 0) + redis_stats.get("misses", 0) > 0:
            hit_rates.append(redis_stats.get("hit_rate", 0))
        if semantic_stats.get("hits", 0) + semantic_stats.get("misses", 0) > 0:
            hit_rates.append(semantic_stats.get("hit_rate", 0))
        if routing_stats.get("hits", 0) + routing_stats.get("misses", 0) > 0:
            hit_rates.append(routing_stats.get("hit_rate", 0))
        
        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
        
        response = CacheStatsResponse(
            redis=redis_stats,
            semantic=semantic_stats,
            routing=routing_stats,
            total_size_mb=round(total_size_mb, 2),
            total_entries=total_entries,
            hit_rate_avg=round(avg_hit_rate * 100, 2),
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "get_cache_stats_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to get cache statistics",
                "request_id": request_id
            }
        )


@router.post(
    "/cache/clear",
    summary="Clear Cache",
    description="""
    Clear all cache entries.
    
    Options:
    - Clear specific cache types (redis, semantic, routing)
    - Clear all caches
    - Pattern-based clearing
    
    Admin only endpoint.
    """,
    response_model=AdminActionResponse,
    dependencies=[Depends(rate_limit(limit=10, period=60))]
)
async def clear_cache(
    request: Request,
    cache_type: str = Query("all", description="Cache type to clear (redis, semantic, routing, all)"),
    pattern: Optional[str] = Query(None, description="Key pattern to clear (redis only)"),
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> AdminActionResponse:
    """
    Clear cache entries.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "clear_cache_requested",
        request_id=request_id,
        admin_id=admin.get("id"),
        cache_type=cache_type,
        pattern=pattern
    )
    
    try:
        cleared_count = 0
        
        if cache_type in ["redis", "all"]:
            if pattern:
                cleared = await cache_manager.clear_pattern(pattern)
            else:
                cleared = await cache_manager.clear_all()
            cleared_count += cleared
        
        if cache_type in ["semantic", "all"]:
            cleared = await semantic_cache.clear()
            cleared_count += cleared
        
        if cache_type in ["routing", "all"]:
            from router.prompt_router import PromptRouter
            router = PromptRouter()
            cleared = await router.clear_cache()
            cleared_count += cleared
        
        response = AdminActionResponse(
            status="cleared",
            message=f"Cache cleared successfully ({cleared_count} entries removed)",
            timestamp=datetime.now(),
            admin_id=admin.get("id"),
            request_id=request_id,
            details={
                "cache_type": cache_type,
                "pattern": pattern,
                "entries_removed": cleared_count
            }
        )
        
        await log_admin_action(
            request=request,
            action="clear_cache",
            resource_type="cache",
            changes={"cache_type": cache_type, "entries_removed": cleared_count},
            status="success"
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "clear_cache_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="clear_cache",
            resource_type="cache",
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to clear cache",
                "request_id": request_id
            }
        )


# ============================================================================
# SYSTEM HEALTH & MAINTENANCE ENDPOINTS
# ============================================================================

@router.get(
    "/health",
    summary="Get System Health",
    description="""
    Get comprehensive system health status.
    
    Returns detailed health information for all components:
    - Database connectivity and performance
    - Redis cache status
    - Model service health
    - GPU availability and utilization
    - Disk usage and capacity
    - Memory usage
    - Active alerts
    
    Admin only endpoint.
    """,
    response_model=SystemHealthResponse,
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_system_health(
    request: Request,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> SystemHealthResponse:
    """
    Get comprehensive system health status.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        # Run all health checks in parallel
        tasks = [
            health_checker.check_all_components(),
            health_checker.check_all_services(),
            health_checker.get_resource_utilization(),
            alert_manager.get_firing_alerts(limit=20)
        ]
        
        components, services, resources, alerts = await asyncio.gather(*tasks)
        
        # Determine overall status
        overall_status = "healthy"
        for component in components.values():
            if component.get("status") == "unhealthy":
                overall_status = "unhealthy"
                break
            elif component.get("status") == "degraded":
                overall_status = "degraded"
        
        # Get uptime
        import psutil
        uptime = time.time() - psutil.boot_time()
        
        response = SystemHealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            components=components,
            services=services,
            resources=resources,
            uptime_seconds=uptime,
            version=settings.version,
            alerts=alerts
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "get_system_health_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to get system health",
                "request_id": request_id
            }
        )


@router.post(
    "/maintenance",
    summary="Set Maintenance Mode",
    description="""
    Enable or disable maintenance mode.
    
    When enabled:
    - All non-admin endpoints return 503
    - Health checks continue to work
    - Admin endpoints remain accessible
    - Scheduled jobs are paused
    
    Admin only endpoint.
    """,
    response_model=AdminActionResponse,
    dependencies=[Depends(rate_limit(limit=5, period=60))]
)
async def set_maintenance_mode(
    request: Request,
    enabled: bool = Query(..., description="Enable maintenance mode"),
    reason: str = Query(None, description="Reason for maintenance"),
    estimated_duration: Optional[int] = Query(None, description="Estimated duration in minutes"),
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> AdminActionResponse:
    """
    Enable or disable maintenance mode.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "maintenance_mode_toggle",
        request_id=request_id,
        admin_id=admin.get("id"),
        enabled=enabled,
        reason=reason
    )
    
    try:
        # Set maintenance mode
        from core.maintenance import MaintenanceManager
        maintenance = MaintenanceManager()
        
        await maintenance.set_maintenance_mode(
            enabled=enabled,
            admin_id=admin.get("id"),
            reason=reason,
            estimated_duration=estimated_duration
        )
        
        response = AdminActionResponse(
            status="maintenance_mode_" + ("enabled" if enabled else "disabled"),
            message=f"Maintenance mode {'enabled' if enabled else 'disabled'}" + 
                   (f": {reason}" if reason else ""),
            timestamp=datetime.now(),
            admin_id=admin.get("id"),
            request_id=request_id,
            details={
                "enabled": enabled,
                "reason": reason,
                "estimated_duration_minutes": estimated_duration
            }
        )
        
        await log_admin_action(
            request=request,
            action="maintenance_mode",
            resource_type="system",
            changes={"enabled": enabled, "reason": reason},
            status="success"
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "maintenance_mode_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="maintenance_mode",
            resource_type="system",
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to set maintenance mode",
                "request_id": request_id
            }
        )


# ============================================================================
# BACKUP & RESTORE ENDPOINTS
# ============================================================================

@router.post(
    "/backup",
    summary="Create Backup",
    description="""
    Create a system backup.
    
    Backs up:
    - Database
    - Configuration files
    - Model metadata
    - Usage statistics
    - Audit logs
    
    Returns backup file for download.
    
    Admin only endpoint.
    """,
    dependencies=[Depends(rate_limit(limit=2, period=3600))]  # 2 per hour
)
async def create_backup(
    request: Request,
    background_tasks: BackgroundTasks,
    include_models: bool = Query(False, description="Include model files (large)"),
    admin: Dict[str, Any] = Depends(verify_admin_access)
):
    """
    Create a system backup.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.info(
        "backup_requested",
        request_id=request_id,
        admin_id=admin.get("id"),
        include_models=include_models
    )
    
    try:
        # Create backup in background
        backup_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            _create_backup_task,
            backup_id=backup_id,
            request_id=request_id,
            admin_id=admin.get("id"),
            include_models=include_models
        )
        
        return {
            "status": "started",
            "backup_id": backup_id,
            "message": "Backup started in background",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(
            "backup_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="create_backup",
            resource_type="system",
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to create backup",
                "request_id": request_id
            }
        )


async def _create_backup_task(
    backup_id: str,
    request_id: str,
    admin_id: str,
    include_models: bool
):
    """
    Background task for creating backup.
    """
    logger.info(
        "backup_task_started",
        backup_id=backup_id,
        request_id=request_id,
        include_models=include_models
    )
    
    try:
        backup_path = await backup_manager.create_backup(
            backup_id=backup_id,
            include_models=include_models
        )
        
        logger.info(
            "backup_task_completed",
            backup_id=backup_id,
            request_id=request_id,
            path=str(backup_path)
        )
        
    except Exception as e:
        logger.error(
            "backup_task_failed",
            backup_id=backup_id,
            request_id=request_id,
            error=str(e),
            exc_info=True
        )


@router.get(
    "/backups",
    summary="List Backups",
    description="""
    List available system backups.
    
    Returns metadata for all backups including size, date, and components.
    
    Admin only endpoint.
    """,
    response_model=List[BackupResponse],
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def list_backups(
    request: Request,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> List[BackupResponse]:
    """
    List available system backups.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        backups = await backup_manager.list_backups()
        
        responses = []
        for backup in backups:
            responses.append(BackupResponse(
                backup_id=backup["id"],
                filename=backup["filename"],
                size_bytes=backup["size_bytes"],
                created_at=backup["created_at"],
                status=backup["status"],
                components=backup["components"]
            ))
        
        return responses
        
    except Exception as e:
        logger.error(
            "list_backups_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to list backups",
                "request_id": request_id
            }
        )


@router.get(
    "/backups/{backup_id}/download",
    summary="Download Backup",
    description="""
    Download a system backup file.
    
    Returns the backup file for download.
    
    Admin only endpoint.
    """,
    dependencies=[Depends(rate_limit(limit=10, period=3600))]  # 10 per hour
)
async def download_backup(
    request: Request,
    backup_id: str,
    admin: Dict[str, Any] = Depends(verify_admin_access)
):
    """
    Download a system backup file.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        backup_path = await backup_manager.get_backup_path(backup_id)
        
        if not backup_path or not backup_path.exists():
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "BackupNotFound",
                    "message": f"Backup '{backup_id}' not found",
                    "request_id": request_id
                }
            )
        
        return FileResponse(
            path=backup_path,
            filename=backup_path.name,
            media_type="application/gzip"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "download_backup_failed",
            request_id=request_id,
            backup_id=backup_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to download backup '{backup_id}'",
                "request_id": request_id
            }
        )


@router.post(
    "/restore/{backup_id}",
    summary="Restore Backup",
    description="""
    Restore system from a backup.
    
    WARNING: This will overwrite current data.
    System will be put in maintenance mode during restore.
    
    Admin only endpoint.
    """,
    response_model=AdminActionResponse,
    dependencies=[Depends(rate_limit(limit=1, period=3600))]  # 1 per hour
)
async def restore_backup(
    request: Request,
    backup_id: str,
    background_tasks: BackgroundTasks,
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> AdminActionResponse:
    """
    Restore system from a backup.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    logger.warning(
        "restore_backup_requested",
        request_id=request_id,
        backup_id=backup_id,
        admin_id=admin.get("id")
    )
    
    try:
        # Check if backup exists
        backup_path = await backup_manager.get_backup_path(backup_id)
        if not backup_path or not backup_path.exists():
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "BackupNotFound",
                    "message": f"Backup '{backup_id}' not found",
                    "request_id": request_id
                }
            )
        
        # Start restore in background
        background_tasks.add_task(
            _restore_backup_task,
            backup_id=backup_id,
            request_id=request_id,
            admin_id=admin.get("id")
        )
        
        # Enable maintenance mode
        from core.maintenance import MaintenanceManager
        maintenance = MaintenanceManager()
        await maintenance.set_maintenance_mode(
            enabled=True,
            admin_id=admin.get("id"),
            reason=f"Restoring from backup {backup_id}"
        )
        
        response = AdminActionResponse(
            status="restore_started",
            message=f"Restore from backup '{backup_id}' started in background",
            timestamp=datetime.now(),
            admin_id=admin.get("id"),
            request_id=request_id,
            details={
                "backup_id": backup_id,
                "estimated_time_seconds": 300  # 5 minutes estimate
            }
        )
        
        await log_admin_action(
            request=request,
            action="restore_backup",
            resource_type="system",
            resource_id=backup_id,
            status="started"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "restore_backup_failed",
            request_id=request_id,
            backup_id=backup_id,
            error=str(e),
            exc_info=True
        )
        
        await log_admin_action(
            request=request,
            action="restore_backup",
            resource_type="system",
            resource_id=backup_id,
            status="error",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to restore backup '{backup_id}'",
                "request_id": request_id
            }
        )


async def _restore_backup_task(
    backup_id: str,
    request_id: str,
    admin_id: str
):
    """
    Background task for restoring backup.
    """
    logger.info(
        "restore_task_started",
        backup_id=backup_id,
        request_id=request_id
    )
    
    try:
        await backup_manager.restore_backup(backup_id)
        
        logger.info(
            "restore_task_completed",
            backup_id=backup_id,
            request_id=request_id
        )
        
        # Disable maintenance mode after restore
        from core.maintenance import MaintenanceManager
        maintenance = MaintenanceManager()
        await maintenance.set_maintenance_mode(
            enabled=False,
            admin_id=admin_id,
            reason="Restore complete"
        )
        
    except Exception as e:
        logger.error(
            "restore_task_failed",
            backup_id=backup_id,
            request_id=request_id,
            error=str(e),
            exc_info=True
        )


# ============================================================================
# AUDIT LOGS ENDPOINTS
# ============================================================================

@router.get(
    "/audit",
    summary="Get Audit Logs",
    description="""
    Get administrative audit logs.
    
    Returns all admin actions with:
    - Who performed the action
    - What action was performed
    - Which resource was affected
    - What changed
    - When it happened
    - IP address and user agent
    
    Supports filtering and pagination.
    
    Admin only endpoint.
    """,
    response_model=Dict[str, Any],
    dependencies=[Depends(rate_limit(limit=30, period=60))]
)
async def get_audit_logs(
    request: Request,
    admin_id: Optional[str] = Query(None, description="Filter by admin user ID"),
    action: Optional[str] = Query(None, description="Filter by action type"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, description="Number of logs to return", ge=1, le=1000),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    admin: Dict[str, Any] = Depends(verify_admin_access)
) -> Dict[str, Any]:
    """
    Get administrative audit logs.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        from database.repositories.audit_repository import AuditRepository
        audit_repo = AuditRepository()
        
        # Set default time range
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=30)
        
        logs = await audit_repo.get_audit_logs(
            admin_id=admin_id,
            action=action,
            resource_type=resource_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )
        
        total = await audit_repo.get_audit_logs_count(
            admin_id=admin_id,
            action=action,
            resource_type=resource_type,
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "logs": logs,
            "filters": {
                "admin_id": admin_id,
                "action": action,
                "resource_type": resource_type,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None
            },
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "get_audit_logs_failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": "Failed to get audit logs",
                "request_id": request_id
            }
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "router",
    "AdminActionResponse",
    "UserCreateRequest",
    "UserUpdateRequest",
    "UserResponse",
    "SystemConfigResponse",
    "SystemConfigUpdateRequest",
    "CacheStatsResponse",
    "SystemHealthResponse",
    "BackupResponse",
    "AuditLogEntry"
]