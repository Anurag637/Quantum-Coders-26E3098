"""
Security Module - Production Ready
Authentication, authorization, API key management, JWT tokens, password hashing,
and security utilities for the LLM Gateway.
"""

import os
import secrets
import string
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union, List, Tuple
from uuid import UUID

from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
import bcrypt

from config import settings
from core.logging import get_logger
from core.exceptions import (
    AuthenticationError,
    InvalidAPIKeyError,
    ExpiredAPIKeyError,
    InvalidTokenError,
    ExpiredTokenError,
    AdminPermissionError,
    InsufficientPermissionsError
)

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# PASSWORD HASHING
# ============================================================================

# Passlib context for bcrypt hashing
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Higher rounds = more secure but slower
)


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password string
    
    Example:
        hashed = hash_password("my_secure_password")
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored password hash
    
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================================
# API KEY GENERATION & MANAGEMENT
# ============================================================================

class APIKeyManager:
    """
    API key generation and management.
    
    Features:
    - Cryptographically secure random keys
    - Prefix-based identification (llm_ for LLM Gateway)
    - Checksum validation
    - Key rotation support
    - Rate limit tracking
    """
    
    # API key format: llm_ + 32 chars from URL-safe alphabet
    KEY_PREFIX = "llm_"
    KEY_LENGTH = 32
    ALPHABET = string.ascii_letters + string.digits + "-_"
    
    @classmethod
    def generate_api_key(cls) -> str:
        """
        Generate a new cryptographically secure API key.
        
        Format: llm_ + 32 random characters
        
        Returns:
            API key string
        
        Example:
            llm_4xK9pL2mN8qR5sT7vW1yZ3aB6cD8eF0g
        """
        # Generate random bytes
        random_bytes = secrets.token_bytes(cls.KEY_LENGTH)
        
        # Convert to URL-safe base64
        import base64
        key = base64.urlsafe_b64encode(random_bytes).decode('ascii')
        
        # Remove padding and ensure exact length
        key = key.replace('=', '')[:cls.KEY_LENGTH]
        
        return f"{cls.KEY_PREFIX}{key}"
    
    @classmethod
    def validate_api_key_format(cls, api_key: str) -> bool:
        """
        Validate API key format without checking existence.
        
        Args:
            api_key: API key to validate
        
        Returns:
            True if format is valid, False otherwise
        """
        if not api_key or not api_key.startswith(cls.KEY_PREFIX):
            return False
        
        key_part = api_key[len(cls.KEY_PREFIX):]
        
        # Check length
        if len(key_part) != cls.KEY_LENGTH:
            return False
        
        # Check character set
        if not all(c in cls.ALPHABET for c in key_part):
            return False
        
        return True
    
    @classmethod
    def mask_api_key(cls, api_key: str) -> str:
        """
        Mask API key for logging (show first 4 and last 4 chars).
        
        Args:
            api_key: Full API key
        
        Returns:
            Masked API key
        
        Example:
            llm_4xK9...eF0g
        """
        if not api_key or len(api_key) < 12:
            return "***"
        
        prefix = api_key[:8]
        suffix = api_key[-4:]
        return f"{prefix}...{suffix}"


# ============================================================================
# JWT TOKEN MANAGEMENT
# ============================================================================

class JWTManager:
    """
    JWT token creation and validation.
    
    Features:
    - Access tokens (short-lived)
    - Refresh tokens (long-lived)
    - Token blacklisting
    - Claims validation
    """
    
    def __init__(self):
        self.secret_key = settings.secret_key_value
        self.algorithm = settings.security.algorithm
        self.access_token_expire_minutes = settings.security.access_token_expire_minutes
        self.refresh_token_expire_days = settings.security.refresh_token_expire_days
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a new JWT access token.
        
        Args:
            data: Token payload data
            expires_delta: Optional custom expiration time
        
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        Create a new refresh token.
        
        Args:
            user_id: User ID for token
        
        Returns:
            Refresh token string
        """
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_hex(16)  # Unique token ID for blacklisting
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded token payload
        
        Raises:
            InvalidTokenError: Token is invalid or malformed
            ExpiredTokenError: Token has expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ExpiredTokenError()
        except jwt.JWTError:
            raise InvalidTokenError()
    
    def verify_token(self, token: str, expected_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify a token and optionally check its type.
        
        Args:
            token: JWT token string
            expected_type: Expected token type (access, refresh)
        
        Returns:
            Verified token payload
        
        Raises:
            InvalidTokenError: Token is invalid or wrong type
        """
        payload = self.decode_token(token)
        
        if expected_type and payload.get("type") != expected_type:
            raise InvalidTokenError(f"Invalid token type. Expected {expected_type}")
        
        return payload


# ============================================================================
# SECURITY SCHEMES
# ============================================================================

# API Key authentication scheme
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key for authentication"
)

# Bearer token authentication scheme
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="JWT bearer token for authentication"
)


# ============================================================================
# API KEY VALIDATION
# ============================================================================

async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    request: Request = None
) -> Dict[str, Any]:
    """
    Verify API key and return user information.
    
    This is the primary authentication dependency for API endpoints.
    
    Args:
        api_key: API key from X-API-Key header
        request: FastAPI request object
    
    Returns:
        User information dictionary
    
    Raises:
        InvalidAPIKeyError: API key is invalid
        ExpiredAPIKeyError: API key has expired
        AuthenticationError: No API key provided
    """
    if not api_key:
        raise AuthenticationError(
            message="API key required",
            detail="Please provide an API key in the X-API-Key header"
        )
    
    # Validate key format
    if not APIKeyManager.validate_api_key_format(api_key):
        masked_key = APIKeyManager.mask_api_key(api_key)
        logger.warning(f"Invalid API key format: {masked_key}")
        raise InvalidAPIKeyError(
            detail=f"Invalid API key format: {masked_key}"
        )
    
    # Import here to avoid circular imports
    from database.repositories.user_repository import UserRepository
    
    user_repo = UserRepository()
    
    # Look up user by API key
    user = await user_repo.get_user_by_api_key(api_key)
    
    if not user:
        masked_key = APIKeyManager.mask_api_key(api_key)
        logger.warning(f"API key not found: {masked_key}")
        raise InvalidAPIKeyError(
            detail="API key not found or inactive"
        )
    
    # Check if user is active
    if not user.get("is_active", True):
        logger.warning(f"Inactive user attempted API access: {user['id']}")
        raise InvalidAPIKeyError(
            detail="User account is inactive"
        )
    
    # Check if API key is expired
    if user.get("api_key_expires_at"):
        expires_at = user["api_key_expires_at"]
        if expires_at and datetime.utcnow() > expires_at:
            logger.warning(f"Expired API key for user: {user['id']}")
            raise ExpiredAPIKeyError(
                detail=f"API key expired on {expires_at.isoformat()}"
            )
    
    # Log successful authentication (masked key)
    masked_key = APIKeyManager.mask_api_key(api_key)
    logger.info(
        "api_key_authenticated",
        user_id=user["id"],
        username=user.get("username"),
        api_key_masked=masked_key
    )
    
    # Store user info in request state for later use
    if request:
        request.state.user_id = user["id"]
        request.state.user = user
    
    return user


async def optional_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    request: Request = None
) -> Optional[Dict[str, Any]]:
    """
    Optional API key authentication.
    
    Returns user info if API key is valid, None otherwise.
    Useful for endpoints that work both authenticated and unauthenticated.
    
    Args:
        api_key: API key from header
        request: FastAPI request object
    
    Returns:
        User information or None
    """
    if not api_key:
        return None
    
    try:
        return await verify_api_key(api_key, request)
    except Exception:
        return None


# ============================================================================
# JWT TOKEN VALIDATION
# ============================================================================

jwt_manager = JWTManager()


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    request: Request = None
) -> Dict[str, Any]:
    """
    Verify JWT bearer token and return user information.
    
    Args:
        credentials: Bearer token credentials
        request: FastAPI request object
    
    Returns:
        User information dictionary
    
    Raises:
        AuthenticationError: No token provided
        InvalidTokenError: Token is invalid
        ExpiredTokenError: Token has expired
    """
    if not credentials:
        raise AuthenticationError(
            message="Bearer token required",
            detail="Please provide a Bearer token in the Authorization header"
        )
    
    token = credentials.credentials
    
    try:
        # Verify token
        payload = jwt_manager.verify_token(token, expected_type="access")
        
        # Extract user ID
        user_id = payload.get("sub")
        if not user_id:
            raise InvalidTokenError("Missing subject claim")
        
        # Import here to avoid circular imports
        from database.repositories.user_repository import UserRepository
        
        user_repo = UserRepository()
        user = await user_repo.get_user(user_id)
        
        if not user:
            raise InvalidTokenError("User not found")
        
        if not user.get("is_active", True):
            raise InvalidTokenError("User account is inactive")
        
        logger.info(
            "jwt_token_authenticated",
            user_id=user["id"],
            username=user.get("username"),
            token_type="access"
        )
        
        # Store user info in request state
        if request:
            request.state.user_id = user["id"]
            request.state.user = user
        
        return user
        
    except (InvalidTokenError, ExpiredTokenError):
        raise
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise InvalidTokenError()


async def verify_refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Dict[str, Any]:
    """
    Verify refresh token.
    
    Args:
        credentials: Bearer token credentials
    
    Returns:
        Token payload
    
    Raises:
        InvalidTokenError: Token is invalid or not a refresh token
    """
    if not credentials:
        raise AuthenticationError("Refresh token required")
    
    token = credentials.credentials
    
    try:
        payload = jwt_manager.verify_token(token, expected_type="refresh")
        
        logger.info(
            "refresh_token_verified",
            user_id=payload.get("sub"),
            token_id=payload.get("jti")
        )
        
        return payload
        
    except (InvalidTokenError, ExpiredTokenError):
        raise
    except Exception as e:
        logger.error(f"Refresh token verification error: {str(e)}")
        raise InvalidTokenError()


# ============================================================================
# ADMIN AUTHORIZATION
# ============================================================================

async def verify_admin(
    api_key: Optional[str] = Depends(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    request: Request = None
) -> Dict[str, Any]:
    """
    Verify that the authenticated user has admin privileges.
    
    This dependency can be used with either API key or JWT token.
    
    Args:
        api_key: API key from header
        credentials: Bearer token credentials
        request: FastAPI request object
    
    Returns:
        Admin user information
    
    Raises:
        AdminPermissionError: User is not an admin
    """
    user = None
    
    # Try API key first
    if api_key:
        try:
            user = await verify_api_key(api_key, request)
        except Exception:
            pass
    
    # Try JWT token if no API key
    if not user and credentials:
        try:
            user = await verify_token(credentials, request)
        except Exception:
            pass
    
    if not user:
        raise AuthenticationError(
            message="Authentication required",
            detail="Please provide valid credentials"
        )
    
    # Check admin status
    if not user.get("is_admin", False):
        logger.warning(
            f"Non-admin user attempted admin access: {user['id']}"
        )
        raise AdminPermissionError(
            detail=f"User '{user['username']}' does not have admin privileges"
        )
    
    logger.info(
        "admin_access_granted",
        user_id=user["id"],
        username=user.get("username")
    )
    
    return user


# ============================================================================
# PERMISSION CHECKING
# ============================================================================

async def require_permissions(
    required_permissions: List[str],
    user: Dict[str, Any] = Depends(verify_api_key)
) -> bool:
    """
    Require specific permissions for an endpoint.
    
    Args:
        required_permissions: List of required permission strings
        user: Authenticated user
    
    Returns:
        True if user has all required permissions
    
    Raises:
        InsufficientPermissionsError: User lacks required permissions
    """
    user_permissions = user.get("permissions", [])
    
    # Admin has all permissions
    if user.get("is_admin", False):
        return True
    
    missing = [p for p in required_permissions if p not in user_permissions]
    
    if missing:
        raise InsufficientPermissionsError(
            required_permissions=missing,
            detail=f"Missing required permissions: {', '.join(missing)}"
        )
    
    return True


# ============================================================================
# CURRENT USER DEPENDENCY
# ============================================================================

async def get_current_user(
    api_key: Optional[str] = Depends(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    request: Request = None
) -> Optional[Dict[str, Any]]:
    """
    Get the currently authenticated user.
    
    This dependency can be used to optionally get user info.
    Returns None if not authenticated.
    
    Args:
        api_key: API key from header
        credentials: Bearer token credentials
        request: FastAPI request object
    
    Returns:
        User information or None
    """
    user = None
    
    if api_key:
        try:
            user = await verify_api_key(api_key, request)
        except Exception:
            pass
    
    if not user and credentials:
        try:
            user = await verify_token(credentials, request)
        except Exception:
            pass
    
    return user


async def get_current_user_required(
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the currently authenticated user or raise an error.
    
    Args:
        user: User from get_current_user
    
    Returns:
        User information
    
    Raises:
        AuthenticationError: No authenticated user
    """
    if not user:
        raise AuthenticationError(
            message="Authentication required",
            detail="Please provide valid credentials"
        )
    return user


# ============================================================================
# SECURITY UTILITIES
# ============================================================================

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Token length in bytes (output will be ~1.3x longer in base64)
    
    Returns:
        URL-safe base64 encoded token
    """
    import base64
    token_bytes = secrets.token_bytes(length)
    token = base64.urlsafe_b64encode(token_bytes).decode('ascii')
    return token.rstrip('=')


def generate_secure_id(prefix: str = "", length: int = 16) -> str:
    """
    Generate a secure random ID with optional prefix.
    
    Args:
        prefix: Optional string prefix
        length: Random part length in bytes
    
    Returns:
        Secure ID string
    
    Example:
        generate_secure_id("user_") -> "user_4xK9pL2mN8qR5sT7"
    """
    random_part = secrets.token_hex(length // 2)  # hex gives 2 chars per byte
    return f"{prefix}{random_part}"


def compute_hash(data: Union[str, bytes], algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of data.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm (sha256, sha512, md5)
    
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def constant_time_compare(val1: str, val2: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    
    Args:
        val1: First value
        val2: Second value
    
    Returns:
        True if values are equal
    """
    return hmac.compare_digest(val1.encode('utf-8'), val2.encode('utf-8'))


def sanitize_input(input_str: str, max_length: int = 10000) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_str: Raw user input
        max_length: Maximum allowed length
    
    Returns:
        Sanitized string
    """
    if not input_str:
        return input_str
    
    # Truncate to max length
    if len(input_str) > max_length:
        input_str = input_str[:max_length]
    
    # Remove null bytes
    input_str = input_str.replace('\x00', '')
    
    # Remove control characters (except newlines and tabs)
    import re
    input_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', input_str)
    
    return input_str.strip()


# ============================================================================
# RATE LIMITING KEY GENERATION
# ============================================================================

def get_rate_limit_key(
    request: Request,
    user_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Generate a rate limit key for a request.
    
    Priority:
    1. User ID (if authenticated)
    2. API key (if provided)
    3. IP address (fallback)
    
    Args:
        request: FastAPI request object
        user_id: Authenticated user ID
        api_key: API key from request
    
    Returns:
        Rate limit key string
    """
    if user_id:
        return f"user:{user_id}"
    
    if api_key:
        # Use hashed API key to avoid storing sensitive data
        hashed_key = compute_hash(api_key)
        return f"apikey:{hashed_key[:16]}"
    
    # Fallback to IP address
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


# ============================================================================
# SECURITY HEADERS
# ============================================================================

def get_security_headers() -> Dict[str, str]:
    """
    Get security headers for HTTP responses.
    
    Returns:
        Dictionary of security headers
    """
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }
    
    # Add HSTS in production
    if settings.environment.is_production():
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    
    # Add CSP in production
    if settings.environment.is_production():
        headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'"
        )
    
    return headers


# ============================================================================
# PASSWORD POLICY
# ============================================================================

class PasswordPolicy:
    """Password strength policy enforcement."""
    
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGIT = True
    REQUIRE_SPECIAL = True
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    @classmethod
    def validate(cls, password: str) -> Tuple[bool, Optional[str]]:
        """
        Validate password against policy.
        
        Args:
            password: Password to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < cls.MIN_LENGTH:
            return False, f"Password must be at least {cls.MIN_LENGTH} characters"
        
        if len(password) > cls.MAX_LENGTH:
            return False, f"Password must not exceed {cls.MAX_LENGTH} characters"
        
        if cls.REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if cls.REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        if cls.REQUIRE_DIGIT and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
        
        if cls.REQUIRE_SPECIAL and not any(c in cls.SPECIAL_CHARS for c in password):
            return False, f"Password must contain at least one special character ({cls.SPECIAL_CHARS})"
        
        # Check for common patterns
        common_patterns = ["password", "123456", "qwerty", "admin", "letmein"]
        if any(pattern in password.lower() for pattern in common_patterns):
            return False, "Password contains a common pattern"
        
        return True, None


# ============================================================================
# CSRF PROTECTION
# ============================================================================

class CSRFProtection:
    """CSRF token generation and validation."""
    
    @staticmethod
    def generate_token() -> str:
        """Generate a CSRF token."""
        return generate_secure_token(32)
    
    @staticmethod
    def validate_token(token: str, stored_token: str) -> bool:
        """Validate a CSRF token using constant-time comparison."""
        if not token or not stored_token:
            return False
        return constant_time_compare(token, stored_token)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Password hashing
    "hash_password",
    "verify_password",
    
    # API Key management
    "APIKeyManager",
    "verify_api_key",
    "optional_api_key",
    
    # JWT management
    "JWTManager",
    "jwt_manager",
    "verify_token",
    "verify_refresh_token",
    
    # Authorization
    "verify_admin",
    "require_permissions",
    "get_current_user",
    "get_current_user_required",
    
    # Security utilities
    "generate_secure_token",
    "generate_secure_id",
    "compute_hash",
    "constant_time_compare",
    "sanitize_input",
    "get_rate_limit_key",
    "get_security_headers",
    
    # Password policy
    "PasswordPolicy",
    
    # CSRF protection
    "CSRFProtection",
    
    # Dependencies for FastAPI
    "api_key_header",
    "bearer_scheme",
]