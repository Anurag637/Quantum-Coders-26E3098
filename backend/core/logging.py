"""
Structured Logging System - Production Ready
Enterprise-grade logging with JSON formatting, context propagation,
request tracing, and integration with monitoring systems.
"""

import json
import sys
import logging
import logging.handlers
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from pythonjsonlogger import jsonlogger
import structlog
from structlog.processors import JSONRenderer, TimeStamper, UnicodeDecoder
import traceback

from config import settings

# ============================================================================
# LOGGER CONSTANTS
# ============================================================================

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Services that should have their own loggers
SERVICE_NAMES = [
    "gateway",
    "api",
    "models",
    "cache",
    "database",
    "routing",
    "monitoring",
    "admin",
    "auth"
]

# Fields that should never be logged
SENSITIVE_FIELDS = {
    "password",
    "api_key",
    "secret",
    "token",
    "authorization",
    "x-api-key",
    "jwt",
    "refresh_token",
    "access_token",
    "credit_card",
    "ssn",
    "passport"
}

# ============================================================================
# CUSTOM JSON LOG FORMATTER
# ============================================================================

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging.
    
    Adds:
    - Timestamp in ISO format
    - Log level name
    - Logger name
    - Process ID
    - Thread ID
    - Request ID (if available)
    - Additional context
    """
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        if not log_record.get("timestamp"):
            now = datetime.utcnow().isoformat() + "Z"
            log_record["timestamp"] = now
        
        # Add log level
        if not log_record.get("level"):
            log_record["level"] = record.levelname
        
        # Add logger name
        if not log_record.get("logger"):
            log_record["logger"] = record.name
        
        # Add process and thread info
        if not log_record.get("process_id"):
            log_record["process_id"] = record.process
        if not log_record.get("thread_id"):
            log_record["thread_id"] = record.thread
        
        # Add file and line number
        if not log_record.get("file"):
            log_record["file"] = record.pathname
        if not log_record.get("line"):
            log_record["line"] = record.lineno
        
        # Add function name
        if not log_record.get("function"):
            log_record["function"] = record.funcName
        
        # Add environment
        if not log_record.get("environment"):
            log_record["environment"] = settings.environment.value
        
        # Add service name
        if not log_record.get("service"):
            log_record["service"] = "llm-gateway"
        
        # Redact sensitive information
        self._redact_sensitive_fields(log_record)
    
    def _redact_sensitive_fields(self, log_record: Dict[str, Any]):
        """Redact sensitive information from log records."""
        for key in list(log_record.keys()):
            # Check if key contains sensitive field name
            if any(sensitive in key.lower() for sensitive in SENSITIVE_FIELDS):
                log_record[key] = "[REDACTED]"
            
            # Recursively redact nested dictionaries
            elif isinstance(log_record[key], dict):
                self._redact_sensitive_fields(log_record[key])
            
            # Redact sensitive patterns in strings
            elif isinstance(log_record[key], str):
                for sensitive in SENSITIVE_FIELDS:
                    if sensitive in key.lower():
                        log_record[key] = "[REDACTED]"
                        break


# ============================================================================
# STRUCTLOG CONFIGURATION
# ============================================================================

def setup_structlog():
    """Configure structlog for structured logging."""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=structlog.threadlocal.wrap_dict(dict),
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# ============================================================================
# LOG CONTEXT MANAGER
# ============================================================================

class LogContext:
    """
    Context manager for adding context to logs within a scope.
    
    Example:
        with LogContext(request_id="abc-123", user_id="user-456"):
            logger.info("Processing request")
    """
    
    def __init__(self, **context):
        self.context = context
    
    def __enter__(self):
        structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        structlog.contextvars.unbind_contextvars(*self.context.keys())


# ============================================================================
# REQUEST LOGGER
# ============================================================================

class RequestLogger:
    """
    Specialized logger for HTTP request logging.
    
    Tracks:
    - Request ID
    - Method and path
    - Client IP
    - User agent
    - Response status
    - Duration
    - User ID (if authenticated)
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_request_start(self, request_id: str, method: str, path: str, 
                          client_ip: str, user_agent: str = None):
        """Log the start of a request."""
        self.logger.info(
            "request_started",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            user_agent=user_agent,
            event_type="request_start"
        )
    
    def log_request_end(self, request_id: str, method: str, path: str,
                        status_code: int, duration_ms: float,
                        user_id: str = None):
        """Log the completion of a request."""
        log_data = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "event_type": "request_end"
        }
        
        if user_id:
            log_data["user_id"] = user_id
        
        # Determine log level based on status code
        if status_code >= 500:
            self.logger.error("request_failed", **log_data)
        elif status_code >= 400:
            self.logger.warning("request_client_error", **log_data)
        else:
            self.logger.info("request_completed", **log_data)
    
    def log_request_error(self, request_id: str, method: str, path: str,
                          error: str, exc_info: bool = False):
        """Log a request error."""
        self.logger.error(
            "request_error",
            request_id=request_id,
            method=method,
            path=path,
            error=error,
            event_type="request_error",
            exc_info=exc_info
        )


# ============================================================================
# MODEL LOGGER
# ============================================================================

class ModelLogger:
    """
    Specialized logger for model inference logging.
    
    Tracks:
    - Model ID
    - Inference latency
    - Token usage
    - Success/failure
    - Cache hits
    - Cost
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_model_load(self, model_id: str, status: str, duration_ms: float = None,
                       memory_mb: float = None, error: str = None):
        """Log model loading events."""
        log_data = {
            "model_id": model_id,
            "status": status,
            "event_type": "model_load"
        }
        
        if duration_ms:
            log_data["duration_ms"] = round(duration_ms, 2)
        
        if memory_mb:
            log_data["memory_mb"] = round(memory_mb, 2)
        
        if error:
            log_data["error"] = error
            self.logger.error("model_load_failed", **log_data)
        elif status == "success":
            self.logger.info("model_loaded", **log_data)
        else:
            self.logger.info("model_load_started", **log_data)
    
    def log_model_inference(self, model_id: str, request_id: str,
                           latency_ms: float, tokens: int = None,
                           success: bool = True, error: str = None,
                           cache_hit: bool = False, cost: float = None):
        """Log model inference events."""
        log_data = {
            "model_id": model_id,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "event_type": "model_inference"
        }
        
        if tokens:
            log_data["tokens"] = tokens
        
        if cache_hit:
            log_data["cache_hit"] = True
        
        if cost:
            log_data["cost_usd"] = round(cost, 6)
        
        if error:
            log_data["error"] = error
            self.logger.error("model_inference_failed", **log_data)
        else:
            self.logger.info("model_inference_completed", **log_data)


# ============================================================================
# DATABASE LOGGER
# ============================================================================

class DatabaseLogger:
    """
    Specialized logger for database operations.
    
    Tracks:
    - Query type (SELECT, INSERT, UPDATE, DELETE)
    - Table name
    - Duration
    - Row count
    - Connection pool status
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_query(self, query_type: str, table: str, duration_ms: float,
                  rows_affected: int = None, error: str = None):
        """Log database query events."""
        log_data = {
            "query_type": query_type,
            "table": table,
            "duration_ms": round(duration_ms, 2),
            "event_type": "database_query"
        }
        
        if rows_affected is not None:
            log_data["rows_affected"] = rows_affected
        
        if error:
            log_data["error"] = error
            self.logger.error("database_query_failed", **log_data)
        else:
            self.logger.debug("database_query_completed", **log_data)
    
    def log_connection_pool(self, pool_size: int, active_connections: int,
                           idle_connections: int, waiting_requests: int):
        """Log database connection pool status."""
        self.logger.info(
            "database_pool_status",
            pool_size=pool_size,
            active_connections=active_connections,
            idle_connections=idle_connections,
            waiting_requests=waiting_requests,
            event_type="database_pool"
        )


# ============================================================================
# CACHE LOGGER
# ============================================================================

class CacheLogger:
    """
    Specialized logger for cache operations.
    
    Tracks:
    - Operation (GET, SET, DELETE)
    - Cache key
    - Hit/miss
    - Latency
    - Size
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_cache_operation(self, operation: str, key: str, hit: bool = None,
                           latency_ms: float = None, size_bytes: int = None,
                           error: str = None):
        """Log cache operation events."""
        log_data = {
            "operation": operation,
            "key": key[:50] + "..." if len(key) > 50 else key,
            "event_type": "cache_operation"
        }
        
        if hit is not None:
            log_data["hit"] = hit
        
        if latency_ms:
            log_data["latency_ms"] = round(latency_ms, 2)
        
        if size_bytes:
            log_data["size_bytes"] = size_bytes
        
        if error:
            log_data["error"] = error
            self.logger.error("cache_operation_failed", **log_data)
        else:
            self.logger.debug("cache_operation_completed", **log_data)


# ============================================================================
# AUDIT LOGGER
# ============================================================================

class AuditLogger:
    """
    Specialized logger for audit events.
    
    Tracks:
    - Admin actions
    - Configuration changes
    - User management
    - Security events
    
    These logs are written to a separate audit log file
    and have higher retention requirements.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
    
    def log_admin_action(self, admin_id: str, admin_username: str,
                        action: str, resource_type: str,
                        resource_id: str = None, changes: Dict[str, Any] = None,
                        ip_address: str = None, status: str = "success",
                        error: str = None):
        """Log administrative actions."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "admin_action",
            "admin_id": admin_id,
            "admin_username": admin_username,
            "action": action,
            "resource_type": resource_type,
            "status": status
        }
        
        if resource_id:
            log_entry["resource_id"] = resource_id
        
        if changes:
            log_entry["changes"] = changes
        
        if ip_address:
            log_entry["ip_address"] = ip_address
        
        if error:
            log_entry["error"] = error
        
        self.logger.info(json.dumps(log_entry))
    
    def log_security_event(self, event_type: str, user_id: str = None,
                          ip_address: str = None, details: Dict[str, Any] = None):
        """Log security-related events."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "security_event",
            "security_event_type": event_type
        }
        
        if user_id:
            log_entry["user_id"] = user_id
        
        if ip_address:
            log_entry["ip_address"] = ip_address
        
        if details:
            log_entry["details"] = details
        
        self.logger.warning(json.dumps(log_entry))


# ============================================================================
# MAIN LOGGER FACTORY
# ============================================================================

class LoggerFactory:
    """
    Factory for creating and managing loggers.
    
    Provides:
    - Consistent logger configuration
    - Service-specific loggers
    - Context propagation
    - File and console handlers
    """
    
    _instance = None
    _loggers = {}
    _audit_logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the logger factory."""
        # Configure root logger
        self._configure_root_logger()
        
        # Setup structlog
        setup_structlog()
        
        # Create audit logger
        self._audit_logger = self._create_audit_logger()
        
        # Create service loggers
        for service in SERVICE_NAMES:
            self._loggers[service] = self._create_service_logger(service)
    
    def _configure_root_logger(self):
        """Configure the root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(LOG_LEVELS.get(settings.log_level.upper(), logging.INFO))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_formatter())
        root_logger.addHandler(console_handler)
        
        # Add file handler if configured
        if settings.logging.file_path:
            file_handler = self._create_file_handler()
            root_logger.addHandler(file_handler)
    
    def _get_formatter(self):
        """Get the appropriate formatter based on configuration."""
        if settings.logging.format == "json":
            return CustomJsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            # Console formatter for development
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _create_file_handler(self):
        """Create a file handler with rotation."""
        log_path = Path(settings.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=settings.logging.max_size_mb * 1024 * 1024,
            backupCount=settings.logging.backup_count
        )
        
        handler.setFormatter(self._get_formatter())
        return handler
    
    def _create_audit_logger(self):
        """Create a separate logger for audit events."""
        logger = logging.getLogger("audit")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        
        # Audit logs go to a separate file
        if settings.logging.file_path:
            audit_path = Path(settings.logging.file_path).parent / "audit.log"
            handler = logging.handlers.RotatingFileHandler(
                filename=audit_path,
                maxBytes=settings.logging.max_size_mb * 1024 * 1024,
                backupCount=settings.logging.backup_count * 2  # Keep more audit logs
            )
            handler.setFormatter(CustomJsonFormatter())
            logger.addHandler(handler)
        
        return logger
    
    def _create_service_logger(self, service_name: str):
        """Create a logger for a specific service."""
        logger = structlog.get_logger(service_name)
        return logger
    
    def get_logger(self, name: str = None) -> structlog.BoundLogger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (defaults to calling module)
        
        Returns:
            Structured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        # Create new logger for custom name
        logger = structlog.get_logger(name or __name__)
        return logger
    
    def get_request_logger(self) -> RequestLogger:
        """Get a request logger instance."""
        return RequestLogger(self.get_logger("gateway"))
    
    def get_model_logger(self) -> ModelLogger:
        """Get a model logger instance."""
        return ModelLogger(self.get_logger("models"))
    
    def get_database_logger(self) -> DatabaseLogger:
        """Get a database logger instance."""
        return DatabaseLogger(self.get_logger("database"))
    
    def get_cache_logger(self) -> CacheLogger:
        """Get a cache logger instance."""
        return CacheLogger(self.get_logger("cache"))
    
    def get_audit_logger(self) -> AuditLogger:
        """Get the audit logger instance."""
        if not self._audit_logger:
            self._audit_logger = AuditLogger()
        return self._audit_logger


# ============================================================================
# GLOBAL LOGGER INSTANCE
# ============================================================================

_logger_factory = LoggerFactory()


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    This is the main function to use throughout the application.
    
    Args:
        name: Logger name (defaults to calling module)
    
    Returns:
        Structured logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("User logged in", user_id=user.id)
    """
    return _logger_factory.get_logger(name)


def get_request_logger() -> RequestLogger:
    """Get a request logger instance."""
    return _logger_factory.get_request_logger()


def get_model_logger() -> ModelLogger:
    """Get a model logger instance."""
    return _logger_factory.get_model_logger()


def get_database_logger() -> DatabaseLogger:
    """Get a database logger instance."""
    return _logger_factory.get_database_logger()


def get_cache_logger() -> CacheLogger:
    """Get a cache logger instance."""
    return _logger_factory.get_cache_logger()


def get_audit_logger() -> AuditLogger:
    """Get the audit logger instance."""
    return _logger_factory.get_audit_logger()


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def log_exception(logger: structlog.BoundLogger, exc: Exception,
                  message: str = "An error occurred", **context):
    """
    Log an exception with full context.
    
    Args:
        logger: Logger instance
        exc: Exception object
        message: Error message
        **context: Additional context
    """
    context["error_type"] = exc.__class__.__name__
    context["error_message"] = str(exc)
    context["traceback"] = traceback.format_exc()
    
    logger.error(message, **context)


def log_performance(logger: structlog.BoundLogger, operation: str,
                   duration_ms: float, **context):
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        **context: Additional context
    """
    context["operation"] = operation
    context["duration_ms"] = round(duration_ms, 2)
    context["event_type"] = "performance"
    
    # Log at INFO level for normal operations
    if duration_ms < 100:
        logger.debug("performance_ok", **context)
    elif duration_ms < 500:
        logger.info("performance_degraded", **context)
    elif duration_ms < 1000:
        logger.warning("performance_slow", **context)
    else:
        logger.error("performance_critical", **context)


def log_security_event(event_type: str, user_id: str = None,
                      ip_address: str = None, **details):
    """
    Log a security event to the audit log.
    
    Args:
        event_type: Type of security event
        user_id: User ID (if applicable)
        ip_address: IP address (if applicable)
        **details: Additional details
    """
    audit_logger = get_audit_logger()
    audit_logger.log_security_event(event_type, user_id, ip_address, details)


def log_admin_action(admin_id: str, admin_username: str, action: str,
                    resource_type: str, resource_id: str = None,
                    changes: Dict[str, Any] = None, ip_address: str = None,
                    status: str = "success", error: str = None):
    """
    Log an administrative action to the audit log.
    
    Args:
        admin_id: Admin user ID
        admin_username: Admin username
        action: Action performed
        resource_type: Type of resource
        resource_id: Resource ID (if applicable)
        changes: Changes made (if applicable)
        ip_address: IP address
        status: Action status
        error: Error message (if failed)
    """
    audit_logger = get_audit_logger()
    audit_logger.log_admin_action(
        admin_id, admin_username, action, resource_type,
        resource_id, changes, ip_address, status, error
    )


# ============================================================================
# INITIALIZATION FUNCTION
# ============================================================================

def setup_logging():
    """
    Initialize the logging system.
    
    This should be called once at application startup.
    
    Example:
        from core.logging import setup_logging, get_logger
        
        setup_logging()
        logger = get_logger(__name__)
    """
    # This ensures the logger factory is initialized
    global _logger_factory
    _logger_factory = LoggerFactory()
    
    # Log startup message
    logger = get_logger("system")
    
    startup_msg = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    LOGGING SYSTEM INITIALIZED                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Environment: {settings.environment.value:<32} ║
    ║  Log Level:   {settings.log_level:<32} ║
    ║  Format:      {settings.logging.format:<32} ║
    ║  File:        {settings.logging.file_path or 'Console':<32} ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    
    logger.info(startup_msg)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main functions
    "setup_logging",
    "get_logger",
    "get_request_logger",
    "get_model_logger",
    "get_database_logger",
    "get_cache_logger",
    "get_audit_logger",
    
    # Logger classes
    "RequestLogger",
    "ModelLogger",
    "DatabaseLogger", 
    "CacheLogger",
    "AuditLogger",
    
    # Utilities
    "log_exception",
    "log_performance",
    "log_security_event",
    "log_admin_action",
    "LogContext",
    
    # Constants
    "LOG_LEVELS",
    "SERVICE_NAMES"
]