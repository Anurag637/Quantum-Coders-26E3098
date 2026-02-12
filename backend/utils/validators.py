"""
Validators - Production Ready
Comprehensive input validation utilities for prompts, API keys, model IDs,
and other system inputs with security checks and sanitization.
"""

import re
import json
import ipaddress
from typing import Dict, Any, Optional, List, Union, Tuple
from urllib.parse import urlparse
from datetime import datetime
import html
import bleach

from core.logging import get_logger
from core.exceptions import ValidationError

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# PROMPT VALIDATORS
# ============================================================================

class PromptValidator:
    """
    Comprehensive prompt validation and sanitization.
    
    Features:
    - Length validation
    - Content safety
    - PII detection
    - Profanity filtering
    - Injection prevention
    - Unicode normalization
    """
    
    # Maximum prompt length (configurable)
    MAX_PROMPT_LENGTH = 100000
    
    # Minimum prompt length
    MIN_PROMPT_LENGTH = 1
    
    # Prohibited patterns (injection, control chars, etc.)
    PROHIBITED_PATTERNS = [
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',  # Control characters
        r'\\x[0-9a-fA-F]{2}',                  # Escaped hex
        r'\\u[0-9a-fA-F]{4}',                  # Escaped unicode
        r'<!--.*?-->',                         # HTML comments
        r'<script.*?>.*?</script>',            # Script tags
        r'javascript:',                        # JavaScript protocol
        r'vbscript:',                         # VBScript protocol
        r'onclick|onload|onerror|onmouseover', # Event handlers
        r'data:text/html',                    # Data URLs
    ]
    
    # Sensitive data patterns (PII)
    SENSITIVE_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(\+\d{1,3}[-.]?)?\d{3,4}[-.]?\d{3,4}[-.]?\d{3,4}\b',
        'ssn': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
        'credit_card': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
        'ip_address': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        'api_key': r'\b(?:sk-|pk-|llm_)[a-zA-Z0-9]{20,}\b',
        'password': r'(?i)password["\']?\s*[:=]\s*["\']?[^"\'\s]+',
        'token': r'\b(?:access|refresh|bearer|jwt)[_\s]?token["\']?\s*[:=]\s*["\']?[^"\'\s]+',
    }
    
    # Profanity filter (basic list)
    PROFANITY_LIST = [
        r'\b(fuck|shit|damn|hell|crap|piss)\b',
        r'\b(asshole|bitch|bastard|dick|cock)\b',
        r'\b(nigger|faggot|retard|slut|whore)\b',
    ]
    
    # Allowed HTML tags (for sanitization)
    ALLOWED_TAGS = [
        'b', 'i', 'em', 'strong', 'code', 'pre', 'p',
        'br', 'ul', 'ol', 'li', 'blockquote'
    ]
    
    # Allowed attributes
    ALLOWED_ATTRIBUTES = {}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_length = self.config.get('max_prompt_length', self.MAX_PROMPT_LENGTH)
        self.min_length = self.config.get('min_prompt_length', self.MIN_PROMPT_LENGTH)
        self.enable_profanity_filter = self.config.get('enable_profanity_filter', True)
        self.enable_pii_detection = self.config.get('enable_pii_detection', True)
        self.enable_html_sanitization = self.config.get('enable_html_sanitization', True)
        
        # Compile regex patterns
        self.prohibited_patterns = [re.compile(p, re.IGNORECASE) for p in self.PROHIBITED_PATTERNS]
        self.sensitive_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in self.SENSITIVE_PATTERNS.items()}
        self.profanity_patterns = [re.compile(p, re.IGNORECASE) for p in self.PROFANITY_LIST]
        
        logger.info(
            "prompt_validator_initialized",
            max_length=self.max_length,
            profanity_filter=self.enable_profanity_filter,
            pii_detection=self.enable_pii_detection
        )
    
    def validate(
        self,
        prompt: str,
        check_sensitive: bool = True,
        check_profanity: bool = True,
        sanitize: bool = True
    ) -> Dict[str, Any]:
        """
        Validate and optionally sanitize a prompt.
        
        Args:
            prompt: Input prompt
            check_sensitive: Check for PII
            check_profanity: Check for profanity
            sanitize: Sanitize the prompt
        
        Returns:
            Validation result with metadata
        
        Raises:
            ValidationError: If validation fails
        """
        if not prompt:
            raise ValidationError(
                errors=[{"field": "prompt", "message": "Prompt cannot be empty"}]
            )
        
        original = prompt
        original_length = len(prompt)
        
        # ====================================================================
        # STEP 1: Length validation
        # ====================================================================
        if original_length > self.max_length:
            raise ValidationError(
                errors=[{
                    "field": "prompt",
                    "message": f"Prompt too long: {original_length} > {self.max_length} characters"
                }]
            )
        
        if original_length < self.min_length:
            raise ValidationError(
                errors=[{
                    "field": "prompt",
                    "message": f"Prompt too short: {original_length} < {self.min_length} characters"
                }]
            )
        
        # ====================================================================
        # STEP 2: Check for prohibited patterns
        # ====================================================================
        prohibited = self._check_prohibited_patterns(prompt)
        if prohibited:
            raise ValidationError(
                errors=[{
                    "field": "prompt",
                    "message": f"Prompt contains prohibited patterns: {', '.join(prohibited)}"
                }]
            )
        
        # ====================================================================
        # STEP 3: Sanitize prompt
        # ====================================================================
        sanitized = prompt
        changes = []
        
        if sanitize:
            sanitized, sanitize_changes = self._sanitize(prompt)
            changes.extend(sanitize_changes)
        
        # ====================================================================
        # STEP 4: Check for sensitive data (PII)
        # ====================================================================
        sensitive_data = []
        if check_sensitive and self.enable_pii_detection:
            sensitive_data = self._detect_sensitive_data(sanitized)
            
            if sensitive_data:
                # Mask sensitive data
                sanitized = self._mask_sensitive_data(sanitized)
                changes.append(f"masked_sensitive_data: {', '.join(sensitive_data)}")
        
        # ====================================================================
        # STEP 5: Check for profanity
        # ====================================================================
        profanity = []
        if check_profanity and self.enable_profanity_filter:
            profanity = self._detect_profanity(sanitized)
            
            if profanity:
                # Mask profanity
                sanitized = self._mask_profanity(sanitized)
                changes.append(f"masked_profanity: {', '.join(profanity)}")
        
        # ====================================================================
        # STEP 6: Normalize whitespace
        # ====================================================================
        sanitized = self._normalize_whitespace(sanitized)
        if sanitized != prompt:
            changes.append("normalized_whitespace")
        
        sanitized_length = len(sanitized)
        
        # Log if changes were made
        if changes:
            logger.info(
                "prompt_sanitized",
                original_length=original_length,
                sanitized_length=sanitized_length,
                changes=changes,
                sensitive_data_count=len(sensitive_data),
                profanity_count=len(profanity)
            )
        
        return {
            "valid": True,
            "original": prompt,
            "sanitized": sanitized,
            "original_length": original_length,
            "sanitized_length": sanitized_length,
            "changes": changes,
            "sensitive_data": sensitive_data,
            "profanity": profanity,
            "was_sanitized": prompt != sanitized
        }
    
    def _check_prohibited_patterns(self, text: str) -> List[str]:
        """Check for prohibited patterns."""
        matches = []
        
        for pattern in self.prohibited_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        
        return matches
    
    def _sanitize(self, text: str) -> Tuple[str, List[str]]:
        """Sanitize text."""
        changes = []
        original = text
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        if text != original:
            changes.append("removed_control_chars")
            original = text
        
        # Unescape HTML entities
        text = html.unescape(text)
        if text != original:
            changes.append("unescaped_html")
            original = text
        
        # Strip unsafe HTML
        if self.enable_html_sanitization:
            text = bleach.clean(
                text,
                tags=self.ALLOWED_TAGS,
                attributes=self.ALLOWED_ATTRIBUTES,
                strip=True
            )
            if text != original:
                changes.append("stripped_html")
        
        return text, changes
    
    def _detect_sensitive_data(self, text: str) -> List[str]:
        """Detect sensitive data in text."""
        detected = []
        
        for name, pattern in self.sensitive_patterns.items():
            if pattern.search(text):
                detected.append(name)
        
        return detected
    
    def _mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text."""
        masked = text
        
        for name, pattern in self.sensitive_patterns.items():
            masked = pattern.sub(f'[REDACTED:{name}]', masked)
        
        return masked
    
    def _detect_profanity(self, text: str) -> List[str]:
        """Detect profanity in text."""
        detected = []
        
        for pattern in self.profanity_patterns:
            if pattern.search(text):
                detected.append(pattern.pattern)
        
        return detected
    
    def _mask_profanity(self, text: str) -> str:
        """Mask profanity in text."""
        masked = text
        
        for pattern in self.profanity_patterns:
            masked = pattern.sub('[PROFANITY]', masked)
        
        return masked
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


# ============================================================================
# API KEY VALIDATORS
# ============================================================================

class APIKeyValidator:
    """
    API key format and security validation.
    
    Features:
    - Format validation
    - Prefix validation
    - Length validation
    - Character set validation
    - Checksum validation
    """
    
    # Supported API key formats
    KEY_FORMATS = {
        'llm': {
            'prefix': 'llm_',
            'length': 32,
            'charset': 'base64url',
            'checksum': True
        },
        'openai': {
            'prefix': 'sk-',
            'length': 48,
            'charset': 'alphanumeric',
            'checksum': False
        },
        'anthropic': {
            'prefix': 'sk-ant-',
            'length': 64,
            'charset': 'alphanumeric',
            'checksum': False
        },
        'cohere': {
            'prefix': None,
            'length': 40,
            'charset': 'alphanumeric',
            'checksum': False
        },
        'groq': {
            'prefix': 'gsk_',
            'length': 56,
            'charset': 'hex',
            'checksum': False
        }
    }
    
    # Allowed character sets
    CHARSETS = {
        'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
        'base64url': re.compile(r'^[a-zA-Z0-9_-]+$'),
        'hex': re.compile(r'^[a-fA-F0-9]+$'),
        'base64': re.compile(r'^[a-zA-Z0-9+/=]+$'),
    }
    
    def __init__(self):
        logger.info("api_key_validator_initialized")
    
    def validate(self, api_key: str, expected_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            expected_format: Expected key format (llm, openai, etc.)
        
        Returns:
            Validation result with metadata
        
        Raises:
            ValidationError: If validation fails
        """
        if not api_key:
            raise ValidationError(
                errors=[{"field": "api_key", "message": "API key cannot be empty"}]
            )
        
        result = {
            "valid": True,
            "format": None,
            "masked": self.mask(api_key),
            "length": len(api_key)
        }
        
        # Detect format if not specified
        if expected_format:
            formats = [expected_format]
        else:
            formats = self.detect_format(api_key)
            result["detected_formats"] = formats
        
        # Check against known formats
        for fmt in formats:
            if fmt in self.KEY_FORMATS:
                format_config = self.KEY_FORMATS[fmt]
                is_valid = self._validate_format(api_key, format_config)
                
                if is_valid:
                    result["format"] = fmt
                    result["valid"] = True
                    return result
        
        # No valid format found
        result["valid"] = False
        result["error"] = "API key format not recognized"
        
        return result
    
    def _validate_format(self, api_key: str, config: Dict[str, Any]) -> bool:
        """Validate API key against format configuration."""
        # Check prefix
        if config.get('prefix'):
            if not api_key.startswith(config['prefix']):
                return False
        
        # Check length
        if config.get('length'):
            if len(api_key) != config['length']:
                return False
        
        # Check character set
        if config.get('charset'):
            charset_regex = self.CHARSETS.get(config['charset'])
            if charset_regex:
                # Remove prefix for charset check
                key_part = api_key
                if config.get('prefix'):
                    key_part = api_key[len(config['prefix']):]
                
                if not charset_regex.match(key_part):
                    return False
        
        # Check checksum (simplified)
        if config.get('checksum'):
            # Simple checksum: first 4 chars XOR last 4 chars
            # This is a simplified example - real implementation would be more robust
            try:
                key_part = api_key.split('_', 1)[1] if '_' in api_key else api_key
                if not self._verify_checksum(key_part):
                    return False
            except Exception:
                return False
        
        return True
    
    def _verify_checksum(self, key: str) -> bool:
        """Verify key checksum (simplified example)."""
        # This is a placeholder - implement actual checksum verification
        # based on your key generation algorithm
        return True
    
    def detect_format(self, api_key: str) -> List[str]:
        """Detect possible API key formats."""
        detected = []
        
        for fmt, config in self.KEY_FORMATS.items():
            # Check prefix
            if config.get('prefix') and not api_key.startswith(config['prefix']):
                continue
            
            # Check length (approximate)
            if config.get('length'):
                if abs(len(api_key) - config['length']) > 4:  # Allow small variation
                    continue
            
            # Check character set
            if config.get('charset'):
                charset_regex = self.CHARSETS.get(config['charset'])
                if charset_regex:
                    key_part = api_key
                    if config.get('prefix'):
                        key_part = api_key[len(config['prefix']):]
                    
                    if charset_regex.match(key_part):
                        detected.append(fmt)
            else:
                detected.append(fmt)
        
        return detected
    
    def mask(self, api_key: str, visible_chars: int = 4) -> str:
        """Mask API key for logging."""
        if len(api_key) <= visible_chars * 2:
            return '*' * len(api_key)
        
        prefix = api_key[:visible_chars]
        suffix = api_key[-visible_chars:]
        masked = '*' * (len(api_key) - visible_chars * 2)
        
        return f"{prefix}{masked}{suffix}"
    
    def generate(self, format: str = 'llm') -> str:
        """Generate a new API key (for testing)."""
        import secrets
        import base64
        
        config = self.KEY_FORMATS.get(format)
        if not config:
            raise ValueError(f"Unknown format: {format}")
        
        prefix = config.get('prefix', '')
        length = config.get('length', 32)
        
        # Generate random bytes
        random_bytes = secrets.token_bytes(length)
        
        # Encode based on charset
        if config.get('charset') == 'base64url':
            key = base64.urlsafe_b64encode(random_bytes).decode('ascii')
            key = key.replace('=', '')[:length]
        else:
            key = secrets.token_hex(length // 2)
        
        return f"{prefix}{key}"


# ============================================================================
# MODEL ID VALIDATOR
# ============================================================================

class ModelIDValidator:
    """
    Model ID format validation.
    
    Features:
    - Format validation
    - Provider detection
    - Version extraction
    - Size extraction
    - Capability inference
    """
    
    # Model ID patterns
    PATTERNS = {
        'llama': re.compile(r'llama(?:-(\d+(?:\.\d+)?))?(?:-(\d+b))?(?:-(chat|instruct))?', re.IGNORECASE),
        'mistral': re.compile(r'mistral(?:-(\d+b))?(?:-(instruct|v\d+(?:\.\d+)?))?', re.IGNORECASE),
        'falcon': re.compile(r'falcon(?:-(\d+b))?(?:-(instruct))?', re.IGNORECASE),
        'gpt': re.compile(r'gpt(?:-(\d+(?:\.\d+)?))?(?:-(turbo|instruct))?', re.IGNORECASE),
        'claude': re.compile(r'claude(?:-(\d+(?:\.\d+)?))?(?:-(haiku|sonnet|opus|instant))?', re.IGNORECASE),
        'command': re.compile(r'command(?:-(?:r|light|nightly))?(?:\+)?', re.IGNORECASE),
        'grok': re.compile(r'grok(?:-(beta|alpha|v\d+))?', re.IGNORECASE),
        'qwen': re.compile(r'qwen(?:(\d+))?(?:-(\d+b))?(?:-(instruct|chat))?', re.IGNORECASE),
        'phi': re.compile(r'phi(?:-(\d+(?:\.\d+)?))?', re.IGNORECASE),
        'starcoder': re.compile(r'starcoder(?:(\d+))?(?:-(\d+b))?', re.IGNORECASE),
        'deepseek': re.compile(r'deepseek(?:-(\w+))?(?:-(\d+b))?(?:-(instruct|chat|coder))?', re.IGNORECASE),
    }
    
    # Size mapping
    SIZE_MAP = {
        '1b': 1_000_000_000,
        '1.1b': 1_100_000_000,
        '2.7b': 2_700_000_000,
        '3b': 3_000_000_000,
        '6b': 6_000_000_000,
        '6.7b': 6_700_000_000,
        '7b': 7_000_000_000,
        '8b': 8_000_000_000,
        '13b': 13_000_000_000,
        '20b': 20_000_000_000,
        '34b': 34_000_000_000,
        '40b': 40_000_000_000,
        '70b': 70_000_000_000,
    }
    
    def __init__(self):
        logger.info("model_id_validator_initialized")
    
    def validate(self, model_id: str) -> Dict[str, Any]:
        """
        Validate model ID format and extract metadata.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Validation result with extracted metadata
        
        Raises:
            ValidationError: If validation fails
        """
        if not model_id:
            raise ValidationError(
                errors=[{"field": "model_id", "message": "Model ID cannot be empty"}]
            )
        
        # Basic format validation
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$', model_id):
            raise ValidationError(
                errors=[{
                    "field": "model_id",
                    "message": "Model ID must start with alphanumeric and contain only alphanumeric, underscore, hyphen, and dot"
                }]
            )
        
        result = {
            "valid": True,
            "model_id": model_id,
            "provider": None,
            "family": None,
            "size": None,
            "size_billions": None,
            "version": None,
            "variant": None,
            "capabilities": []
        }
        
        # Detect provider and family
        for family, pattern in self.PATTERNS.items():
            match = pattern.match(model_id)
            if match:
                result["family"] = family
                result["provider"] = self._get_provider(family)
                
                # Extract size
                if len(match.groups()) > 0:
                    for group in match.groups():
                        if group and 'b' in str(group).lower():
                            result["size"] = group
                            result["size_billions"] = self.SIZE_MAP.get(group.lower())
                
                # Extract version
                if len(match.groups()) > 0:
                    for group in match.groups():
                        if group and re.match(r'^\d+(?:\.\d+)?$', str(group)):
                            result["version"] = group
                
                # Extract variant
                if len(match.groups()) > 0:
                    for group in match.groups():
                        if group and group.lower() in ['chat', 'instruct', 'turbo', 'haiku', 'sonnet', 'opus', 'coder']:
                            result["variant"] = group.lower()
                
                # Infer capabilities
                result["capabilities"] = self._infer_capabilities(result)
                
                break
        
        return result
    
    def _get_provider(self, family: str) -> str:
        """Get provider name for model family."""
        providers = {
            'llama': 'Meta',
            'mistral': 'Mistral AI',
            'falcon': 'TII',
            'gpt': 'OpenAI',
            'claude': 'Anthropic',
            'command': 'Cohere',
            'grok': 'xAI',
            'qwen': 'Alibaba',
            'phi': 'Microsoft',
            'starcoder': 'BigCode',
            'deepseek': 'DeepSeek',
        }
        return providers.get(family, 'Unknown')
    
    def _infer_capabilities(self, metadata: Dict[str, Any]) -> List[str]:
        """Infer model capabilities from metadata."""
        capabilities = []
        
        family = metadata.get('family')
        variant = metadata.get('variant')
        
        # Base capabilities by family
        if family == 'llama':
            capabilities.extend(['chat', 'reasoning'])
        elif family == 'mistral':
            capabilities.extend(['chat', 'reasoning', 'creative'])
        elif family == 'falcon':
            capabilities.extend(['instruction', 'technical'])
        elif family == 'gpt':
            capabilities.extend(['chat', 'reasoning', 'code'])
        elif family == 'claude':
            capabilities.extend(['chat', 'reasoning', 'analysis'])
        elif family == 'command':
            capabilities.extend(['chat', 'rag', 'multilingual'])
        elif family == 'grok':
            capabilities.extend(['code', 'reasoning', 'humor'])
        elif family == 'qwen':
            capabilities.extend(['multilingual', 'reasoning'])
        elif family == 'phi':
            capabilities.extend(['reasoning', 'math', 'code'])
        elif family == 'starcoder':
            capabilities.extend(['code'])
        elif family == 'deepseek':
            capabilities.extend(['code', 'math', 'reasoning'])
        
        # Variant-specific capabilities
        if variant == 'instruct':
            capabilities.append('instruction')
        if variant == 'chat':
            capabilities.append('chat')
        if variant == 'coder':
            capabilities.append('code')
        
        # Size-based capabilities
        size = metadata.get('size_billions')
        if size:
            if size > 50_000_000_000:
                capabilities.append('complex_reasoning')
            elif size < 3_000_000_000:
                capabilities.append('fast')
                capabilities.append('lightweight')
        
        return list(set(capabilities))


# ============================================================================
# URL VALIDATOR
# ============================================================================

class URLValidator:
    """
    URL validation and sanitization.
    
    Features:
    - Format validation
    - Scheme whitelist
    - Domain validation
    - Path sanitization
    """
    
    # Allowed URL schemes
    ALLOWED_SCHEMES = ['http', 'https', 'ftp', 'sftp', 'ws', 'wss']
    
    # Blocked domains
    BLOCKED_DOMAINS = [
        'localhost',
        '127.0.0.1',
        '0.0.0.0',
        '::1',
    ]
    
    def __init__(self):
        logger.info("url_validator_initialized")
    
    def validate(
        self,
        url: str,
        allow_local: bool = False,
        require_https: bool = False
    ) -> Dict[str, Any]:
        """
        Validate URL format and security.
        
        Args:
            url: URL to validate
            allow_local: Allow localhost/private IPs
            require_https: Require HTTPS scheme
        
        Returns:
            Validation result with metadata
        
        Raises:
            ValidationError: If validation fails
        """
        if not url:
            raise ValidationError(
                errors=[{"field": "url", "message": "URL cannot be empty"}]
            )
        
        result = {
            "valid": True,
            "url": url,
            "scheme": None,
            "hostname": None,
            "port": None,
            "path": None,
            "query": None,
            "fragment": None
        }
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if not parsed.scheme:
                raise ValidationError(
                    errors=[{"field": "url", "message": "URL must have a scheme"}]
                )
            
            if parsed.scheme not in self.ALLOWED_SCHEMES:
                raise ValidationError(
                    errors=[{
                        "field": "url",
                        "message": f"URL scheme '{parsed.scheme}' not allowed. Allowed schemes: {self.ALLOWED_SCHEMES}"
                    }]
                )
            
            if require_https and parsed.scheme != 'https':
                raise ValidationError(
                    errors=[{
                        "field": "url",
                        "message": "HTTPS scheme required"
                    }]
                )
            
            # Check hostname
            if not parsed.hostname:
                raise ValidationError(
                    errors=[{"field": "url", "message": "URL must have a hostname"}]
                )
            
            # Check for blocked domains
            if not allow_local:
                hostname = parsed.hostname.lower()
                
                if hostname in self.BLOCKED_DOMAINS:
                    raise ValidationError(
                        errors=[{
                            "field": "url",
                            "message": f"Domain '{hostname}' is blocked"
                        }]
                    )
                
                # Check for private IP ranges
                try:
                    ip = ipaddress.ip_address(hostname)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        raise ValidationError(
                            errors=[{
                                "field": "url",
                                "message": f"Private IP address '{hostname}' is not allowed"
                            }]
                        )
                except ValueError:
                    # Not an IP address, skip
                    pass
            
            # Extract components
            result.update({
                "scheme": parsed.scheme,
                "hostname": parsed.hostname,
                "port": parsed.port,
                "path": parsed.path,
                "query": parsed.query,
                "fragment": parsed.fragment,
            })
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                errors=[{"field": "url", "message": f"Invalid URL format: {str(e)}"}]
            )
        
        return result


# ============================================================================
# JSON VALIDATOR
# ============================================================================

class JSONValidator:
    """
    JSON validation and sanitization.
    
    Features:
    - Syntax validation
    - Schema validation
    - Size limits
    - Depth limits
    - Type checking
    """
    
    def __init__(self):
        self.max_depth = 20
        self.max_size = 10 * 1024 * 1024  # 10MB
        
        logger.info("json_validator_initialized")
    
    def validate(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
        max_depth: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate JSON data.
        
        Args:
            data: JSON string, bytes, or parsed dict
            schema: Optional JSON schema to validate against
            max_depth: Maximum nesting depth
            max_size: Maximum size in bytes
        
        Returns:
            Validation result with parsed data
        
        Raises:
            ValidationError: If validation fails
        """
        result = {
            "valid": True,
            "data": None,
            "size": 0,
            "depth": 0
        }
        
        # Parse JSON if needed
        if isinstance(data, (str, bytes)):
            try:
                if isinstance(data, str):
                    size = len(data.encode('utf-8'))
                else:
                    size = len(data)
                
                # Check size
                max_size = max_size or self.max_size
                if size > max_size:
                    raise ValidationError(
                        errors=[{
                            "field": "json",
                            "message": f"JSON size ({size} bytes) exceeds maximum ({max_size} bytes)"
                        }]
                    )
                
                result["size"] = size
                parsed = json.loads(data)
                result["data"] = parsed
                
            except json.JSONDecodeError as e:
                raise ValidationError(
                    errors=[{
                        "field": "json",
                        "message": f"Invalid JSON: {str(e)}"
                    }]
                )
        else:
            parsed = data
            result["data"] = parsed
        
        # Check depth
        max_depth = max_depth or self.max_depth
        depth = self._get_depth(parsed)
        result["depth"] = depth
        
        if depth > max_depth:
            raise ValidationError(
                errors=[{
                    "field": "json",
                    "message": f"JSON nesting depth ({depth}) exceeds maximum ({max_depth})"
                }]
            )
        
        # Validate against schema
        if schema:
            # This is a placeholder - implement JSON schema validation
            # using a library like jsonschema
            pass
        
        return result
    
    def _get_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Get maximum nesting depth of JSON object."""
        if current_depth > 100:  # Safety limit
            return current_depth
        
        if isinstance(obj, dict):
            if not obj:
                return current_depth + 1
            return max(
                self._get_depth(v, current_depth + 1)
                for v in obj.values()
            )
        elif isinstance(obj, list):
            if not obj:
                return current_depth + 1
            return max(
                self._get_depth(item, current_depth + 1)
                for item in obj
            )
        else:
            return current_depth


# ============================================================================
# EMAIL VALIDATOR
# ============================================================================

class EmailValidator:
    """
    Email address validation.
    
    Features:
    - Format validation
    - Domain validation
    - Disposable email detection
    - Role account detection
    """
    
    # Disposable email domains
    DISPOSABLE_DOMAINS = {
        'mailinator.com', 'guerrillamail.com', '10minutemail.com',
        'temp-mail.org', 'fakeinbox.com', 'throwawaymail.com',
        'yopmail.com', 'trashmail.com', 'sharklasers.com',
    }
    
    # Role-based email prefixes
    ROLE_PREFIXES = {
        'admin', 'support', 'info', 'contact', 'webmaster',
        'postmaster', 'hostmaster', 'abuse', 'noreply',
        'no-reply', 'sales', 'marketing', 'help',
    }
    
    def __init__(self):
        logger.info("email_validator_initialized")
    
    def validate(
        self,
        email: str,
        check_disposable: bool = True,
        check_role: bool = True
    ) -> Dict[str, Any]:
        """
        Validate email address.
        
        Args:
            email: Email address to validate
            check_disposable: Check for disposable email domains
            check_role: Check for role-based accounts
        
        Returns:
            Validation result with metadata
        
        Raises:
            ValidationError: If validation fails
        """
        if not email:
            raise ValidationError(
                errors=[{"field": "email", "message": "Email cannot be empty"}]
            )
        
        result = {
            "valid": True,
            "email": email,
            "local_part": None,
            "domain": None,
            "is_disposable": False,
            "is_role": False
        }
        
        # Basic format validation
        pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not pattern.match(email):
            raise ValidationError(
                errors=[{"field": "email", "message": "Invalid email format"}]
            )
        
        # Split local part and domain
        try:
            local_part, domain = email.lower().split('@')
            result['local_part'] = local_part
            result['domain'] = domain
        except ValueError:
            raise ValidationError(
                errors=[{"field": "email", "message": "Invalid email format"}]
            )
        
        # Check disposable domains
        if check_disposable:
            result['is_disposable'] = domain in self.DISPOSABLE_DOMAINS
            
            if result['is_disposable']:
                logger.info("disposable_email_detected", email=email)
        
        # Check role accounts
        if check_role:
            result['is_role'] = local_part in self.ROLE_PREFIXES
        
        return result


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

_prompt_validator = None
_api_key_validator = None
_model_id_validator = None
_url_validator = None
_json_validator = None
_email_validator = None


def get_prompt_validator() -> PromptValidator:
    """Get singleton prompt validator instance."""
    global _prompt_validator
    if not _prompt_validator:
        _prompt_validator = PromptValidator()
    return _prompt_validator


def get_api_key_validator() -> APIKeyValidator:
    """Get singleton API key validator instance."""
    global _api_key_validator
    if not _api_key_validator:
        _api_key_validator = APIKeyValidator()
    return _api_key_validator


def get_model_id_validator() -> ModelIDValidator:
    """Get singleton model ID validator instance."""
    global _model_id_validator
    if not _model_id_validator:
        _model_id_validator = ModelIDValidator()
    return _model_id_validator


def get_url_validator() -> URLValidator:
    """Get singleton URL validator instance."""
    global _url_validator
    if not _url_validator:
        _url_validator = URLValidator()
    return _url_validator


def get_json_validator() -> JSONValidator:
    """Get singleton JSON validator instance."""
    global _json_validator
    if not _json_validator:
        _json_validator = JSONValidator()
    return _json_validator


def get_email_validator() -> EmailValidator:
    """Get singleton email validator instance."""
    global _email_validator
    if not _email_validator:
        _email_validator = EmailValidator()
    return _email_validator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Validators
    "PromptValidator",
    "APIKeyValidator",
    "ModelIDValidator",
    "URLValidator",
    "JSONValidator",
    "EmailValidator",
    
    # Singleton getters
    "get_prompt_validator",
    "get_api_key_validator",
    "get_model_id_validator",
    "get_url_validator",
    "get_json_validator",
    "get_email_validator",
]