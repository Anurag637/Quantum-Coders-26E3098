"""
Prompt Analyzer - Production Ready
Advanced NLP analysis for prompt classification, complexity scoring,
language detection, entity extraction, and capability requirements.
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import Counter
import math

from core.logging import get_logger
from core.exceptions import ValidationError
from config import settings

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# PROMPT TYPE CLASSIFICATION
# ============================================================================

class PromptTypeClassifier:
    """
    Classify prompts into semantic categories.
    
    Categories:
    - code: Programming, debugging, algorithms
    - qa: Question answering, facts, explanations
    - creative: Stories, poems, creative writing
    - reasoning: Logic, math, problem solving
    - translation: Language translation
    - summarization: Text summarization
    - analysis: Data analysis, critique
    - instruction: Step-by-step guidance
    - conversation: General chat, dialogue
    - technical: Technical documentation, specs
    """
    
    # Keywords by category
    CATEGORY_KEYWORDS = {
        "code": [
            "python", "javascript", "java", "c++", "function", "class",
            "algorithm", "debug", "compile", "syntax", "api", "endpoint",
            "array", "list", "dictionary", "loop", "conditional", "variable",
            "import", "def ", "return ", "print", "console.log", "React",
            "Django", "Flask", "SQL", "database", "query", "SELECT", "INSERT",
            "git", "commit", "branch", "merge", "docker", "kubernetes",
            "refactor", "optimize", "performance", "bug", "error", "exception"
        ],
        "qa": [
            "what is", "who is", "when did", "where is", "why does",
            "how do", "can you explain", "define", "meaning of",
            "difference between", "compare", "versus", "vs",
            "example of", "definition", "purpose of", "benefits of",
            "disadvantages of", "pros and cons", "history of"
        ],
        "creative": [
            "story", "poem", "essay", "fiction", "novel", "character",
            "plot", "setting", "dialogue", "narrative", "describe",
            "imagine", "creative", "write about", "tell me a story",
            "once upon a time", "fantasy", "sci-fi", "mystery",
            "romance", "adventure", "drama", "comedy", "tragedy"
        ],
        "reasoning": [
            "solve", "calculate", "compute", "equation", "formula",
            "math", "algebra", "geometry", "calculus", "statistics",
            "probability", "logic", "deduce", "infer", "conclude",
            "hypothesis", "theory", "proof", "theorem", "puzzle",
            "riddle", "brain teaser", "problem solving"
        ],
        "translation": [
            "translate", "translation", "in french", "in spanish",
            "in german", "in chinese", "in japanese", "in korean",
            "in russian", "in arabic", "in hindi", "convert to",
            "say in", "how do you say", "meaning in", "equivalent in"
        ],
        "summarization": [
            "summarize", "summary", "tl;dr", "in short", "briefly",
            "condense", "abstract", "key points", "main ideas",
            "overview", "recap", "synopsis", "digest", "roundup"
        ],
        "analysis": [
            "analyze", "analysis", "evaluate", "critique", "review",
            "assess", "examine", "interpret", "break down",
            "sentiment", "tone", "style", "patterns", "trends",
            "insights", "recommendations", "strengths", "weaknesses"
        ],
        "instruction": [
            "how to", "steps to", "guide", "tutorial", "walkthrough",
            "instructions", "directions", "recipe", "manual",
            "learn", "teach", "explain step by step", "procedure",
            "method", "technique", "approach", "best practices"
        ],
        "conversation": [
            "hello", "hi", "hey", "how are you", "what's up",
            "nice to meet you", "thanks", "thank you", "you're welcome",
            "good morning", "good afternoon", "good evening",
            "chat", "talk", "discuss", "conversation"
        ],
        "technical": [
            "documentation", "specification", "architecture",
            "design pattern", "system design", "infrastructure",
            "protocol", "standard", "compliance", "certification",
            "technical debt", "scalability", "reliability",
            "availability", "latency", "throughput", "consistency"
        ]
    }
    
    # Negation words that indicate opposite category
    NEGATIONS = ["not", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't"]
    
    def __init__(self):
        # Compile regex patterns for each category
        self.category_patterns = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            # Create pattern that matches whole words
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.category_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def classify(self, prompt: str) -> Dict[str, float]:
        """
        Classify prompt into categories with confidence scores.
        
        Args:
            prompt: Input prompt
        
        Returns:
            Dictionary mapping category to confidence score (0-1)
        """
        prompt_lower = prompt.lower()
        words = set(re.findall(r'\b[a-z]+\b', prompt_lower))
        
        scores = {}
        total_matches = 0
        
        # Count matches per category
        for category, pattern in self.category_patterns.items():
            matches = pattern.findall(prompt_lower)
            if matches:
                # Weight by number of matches and uniqueness
                unique_matches = len(set(matches))
                scores[category] = unique_matches
                total_matches += unique_matches
        
        # Normalize scores
        if total_matches > 0:
            for category in scores:
                scores[category] = scores[category] / total_matches
        
        # Apply negation handling
        for negation in self.NEGATIONS:
            if negation in prompt_lower:
                # Reduce confidence for all categories
                for category in scores:
                    scores[category] *= 0.7
        
        # If no categories matched, default to conversation
        if not scores:
            scores["conversation"] = 0.8
            scores["qa"] = 0.2
        
        # Sort by score descending
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    
    def get_primary_type(self, scores: Dict[str, float]) -> str:
        """Get the primary prompt type."""
        if not scores:
            return "conversation"
        return next(iter(scores.keys()))
    
    def get_secondary_type(self, scores: Dict[str, float]) -> Optional[str]:
        """Get the secondary prompt type if confidence is significant."""
        if len(scores) < 2:
            return None
        
        items = list(scores.items())
        if items[1][1] > 0.3:  # Secondary confidence > 30%
            return items[1][0]
        
        return None


# ============================================================================
# COMPLEXITY SCORER
# ============================================================================

class ComplexityScorer:
    """
    Calculate prompt complexity score based on multiple factors.
    
    Factors:
    - Length: Longer prompts are more complex
    - Vocabulary: Rare words indicate complexity
    - Syntax: Complex sentence structures
    - Domain: Technical/scientific topics
    - Reasoning: Logical steps required
    """
    
    def __init__(self):
        # Common simple words (low complexity)
        self.simple_words = set([
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "i", "you", "he", "she", "it", "we", "they",
            "this", "that", "these", "those",
            "and", "or", "but", "so", "because",
            "what", "where", "when", "why", "who", "how",
            "yes", "no", "maybe", "please", "thank", "hello", "hi"
        ])
        
        # Complex domain indicators
        self.complex_domains = {
            "quantum", "relativity", "thermodynamics", "neural", "algorithm",
            "complexity", "optimization", "topology", "calculus", "differential",
            "encryption", "cryptography", "blockchain", "compiler", "semantics",
            "philosophical", "metaphysics", "epistemology", "phenomenology"
        }
        
        # Reasoning indicators
        self.reasoning_indicators = [
            "why", "how", "explain", "reason", "because", "therefore",
            "conclude", "deduce", "infer", "imply", "contradict", "hypothesis",
            "theorem", "proof", "validate", "verify", "demonstrate"
        ]
    
    def calculate(self, prompt: str) -> Dict[str, Any]:
        """
        Calculate comprehensive complexity score.
        
        Returns:
            Dict with overall score and component scores
        """
        prompt_lower = prompt.lower()
        words = re.findall(r'\b[a-z]+\b', prompt_lower)
        sentences = re.split(r'[.!?]+', prompt)
        
        # Length factor (0-0.3)
        length = len(prompt)
        length_score = min(length / 1000, 0.3)  # Cap at 1000 chars
        
        # Vocabulary factor (0-0.2)
        unique_words = len(set(words))
        total_words = len(words)
        vocabulary_richness = unique_words / max(total_words, 1)
        vocabulary_score = vocabulary_richness * 0.2
        
        # Simple word penalty
        simple_word_count = sum(1 for w in words if w in self.simple_words)
        simple_word_ratio = simple_word_count / max(total_words, 1)
        vocabulary_score *= (1 - simple_word_ratio * 0.5)
        
        # Sentence structure factor (0-0.2)
        avg_sentence_length = total_words / max(len(sentences), 1)
        structure_score = min(avg_sentence_length / 50, 0.2)  # Cap at 50 words/sentence
        
        # Domain complexity factor (0-0.15)
        domain_score = 0
        for word in words:
            if word in self.complex_domains:
                domain_score += 0.03
        domain_score = min(domain_score, 0.15)
        
        # Reasoning requirement factor (0-0.15)
        reasoning_score = 0
        for indicator in self.reasoning_indicators:
            if indicator in prompt_lower:
                reasoning_score += 0.05
        reasoning_score = min(reasoning_score, 0.15)
        
        # Calculate total score (0-1)
        total_score = length_score + vocabulary_score + structure_score + domain_score + reasoning_score
        
        # Determine complexity level
        if total_score < 0.2:
            level = "simple"
        elif total_score < 0.4:
            level = "moderate"
        elif total_score < 0.6:
            level = "complex"
        else:
            level = "very_complex"
        
        return {
            "score": round(total_score, 3),
            "level": level,
            "components": {
                "length": round(length_score, 3),
                "vocabulary": round(vocabulary_score, 3),
                "structure": round(structure_score, 3),
                "domain": round(domain_score, 3),
                "reasoning": round(reasoning_score, 3)
            },
            "metrics": {
                "characters": length,
                "words": total_words,
                "unique_words": unique_words,
                "sentences": len(sentences),
                "avg_sentence_length": round(avg_sentence_length, 1)
            }
        }


# ============================================================================
# LANGUAGE DETECTOR
# ============================================================================

class LanguageDetector:
    """
    Detect the primary language of the prompt.
    Supports major languages with confidence scores.
    """
    
    # Language signatures (common words)
    LANGUAGE_SIGNATURES = {
        "en": {
            "the", "and", "is", "in", "to", "it", "you", "that", "was", "for",
            "are", "with", "as", "his", "they", "be", "at", "one", "have", "this"
        },
        "es": {
            "el", "la", "los", "las", "de", "que", "y", "en", "a", "por",
            "con", "para", "es", "como", "está", "puede", "tiene", "su", "al", "del"
        },
        "fr": {
            "le", "la", "les", "de", "des", "et", "en", "un", "une", "est",
            "sont", "avec", "pour", "dans", "par", "sur", "peut", "fait", "être", "avoir"
        },
        "de": {
            "der", "die", "das", "und", "ist", "mit", "von", "für", "auf", "den",
            "eine", "einer", "dem", "des", "sie", "wir", "ich", "nicht", "auch", "sich"
        },
        "zh": {
            "的", "是", "在", "了", "和", "有", "这", "对", "也", "会",
            "说", "就", "为", "而", "从", "到", "以", "与", "将", "很"
        },
        "ja": {
            "は", "が", "の", "に", "を", "です", "ます", "した", "この", "その",
            "あの", "私", "あなた", "彼", "彼女", "私たち", "彼ら", "ある", "いる", "こと"
        },
        "hi": {
            "है", "और", "का", "के", "में", "से", "को", "यह", "एक", "था",
            "थे", "कर", "सकता", "सकते", "वह", "वे", "मैं", "आप", "हम", "ये"
        }
    }
    
    def __init__(self):
        # Compile signatures for faster lookup
        self.signatures = self.LANGUAGE_SIGNATURES
    
    def detect(self, prompt: str) -> Dict[str, Any]:
        """
        Detect the primary language of the prompt.
        
        Returns:
            Dict with language code, confidence, and alternatives
        """
        prompt_lower = prompt.lower()
        words = set(re.findall(r'\b[a-z]+\b', prompt_lower))
        
        # Count non-English script characters
        non_english_chars = sum(1 for c in prompt if ord(c) > 127)
        non_english_ratio = non_english_chars / max(len(prompt), 1)
        
        # If >50% non-ASCII, likely non-English
        if non_english_ratio > 0.5:
            # Detect CJK
            cjk_chars = sum(1 for c in prompt if '\u4e00' <= c <= '\u9fff')
            if cjk_chars > 0:
                return {
                    "code": "zh",
                    "name": "Chinese",
                    "confidence": 0.9,
                    "is_primary": True
                }
            
            # Detect Japanese (Hiragana, Katakana)
            hiragana = sum(1 for c in prompt if '\u3040' <= c <= '\u309f')
            katakana = sum(1 for c in prompt if '\u30a0' <= c <= '\u30ff')
            if hiragana > 0 or katakana > 0:
                return {
                    "code": "ja",
                    "name": "Japanese",
                    "confidence": 0.9,
                    "is_primary": True
                }
            
            # Detect Devanagari (Hindi, Sanskrit)
            devanagari = sum(1 for c in prompt if '\u0900' <= c <= '\u097f')
            if devanagari > 0:
                return {
                    "code": "hi",
                    "name": "Hindi",
                    "confidence": 0.9,
                    "is_primary": True
                }
        
        # Language scoring based on common words
        scores = {}
        for lang, signature in self.signatures.items():
            matches = words.intersection(signature)
            score = len(matches) / max(len(signature), 1)
            if score > 0:
                scores[lang] = score
        
        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_scores:
            return {
                "code": "en",
                "name": "English",
                "confidence": 0.5,
                "is_primary": True
            }
        
        primary_lang = sorted_scores[0]
        confidence = primary_lang[1]
        
        # Boost confidence if multiple matches
        if len(sorted_scores) > 1:
            confidence = min(confidence + 0.1, 1.0)
        
        return {
            "code": primary_lang[0],
            "name": self._get_language_name(primary_lang[0]),
            "confidence": round(confidence, 2),
            "is_primary": True,
            "alternatives": [
                {"code": lang, "name": self._get_language_name(lang), "score": round(score, 2)}
                for lang, score in sorted_scores[1:3]
            ] if len(sorted_scores) > 1 else []
        }
    
    def _get_language_name(self, code: str) -> str:
        """Get full language name from code."""
        names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ja": "Japanese",
            "hi": "Hindi"
        }
        return names.get(code, code)


# ============================================================================
# ENTITY EXTRACTOR
# ============================================================================

class EntityExtractor:
    """
    Extract entities from prompts using pattern matching.
    
    Entity types:
    - CODE: Programming languages, frameworks
    - TECH: Technologies, tools
    - DOMAIN: Subject domains
    - PERSON: People names
    - ORG: Organizations
    - LOCATION: Places
    - DATE: Temporal expressions
    - NUMBER: Numerical values
    """
    
    # Entity patterns
    ENTITY_PATTERNS = {
        "code_language": [
            r'\b(python|javascript|java|c\+\+|ruby|php|swift|kotlin|typescript|go|rust|scala)\b',
            r'\b(html|css|sql|regex|bash|powershell|graphql)\b'
        ],
        "framework": [
            r'\b(react|angular|vue|django|flask|spring|rails|express|tensorflow|pytorch)\b',
            r'\b(jquery|bootstrap|tailwind|sass|less|webpack|babel|eslint|jest)\b'
        ],
        "technology": [
            r'\b(docker|kubernetes|aws|azure|gcp|git|jenkins|terraform|ansible)\b',
            r'\b(nginx|apache|redis|postgresql|mysql|mongodb|elasticsearch|kafka)\b'
        ],
        "domain": [
            r'\b(machine\s*learning|ai|artificial\s*intelligence|deep\s*learning|nlp)\b',
            r'\b(blockchain|cryptocurrency|web3|fintech|insurtech|medtech|edtech)\b',
            r'\b(cybersecurity|cloud\s*computing|devops|serverless|microservices)\b'
        ],
        "person": [
            r'\b(Dr\.?\s+[A-Z][a-z]+|Prof\.?\s+[A-Z][a-z]+|Mr\.?\s+[A-Z][a-z]+|Ms\.?\s+[A-Z][a-z]+)\b',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'  # Simple first last name
        ],
        "organization": [
            r'\b(Google|Microsoft|Amazon|Apple|Meta|Netflix|Uber|Airbnb|Salesforce|Oracle|IBM)\b',
            r'\b(Stanford|MIT|Harvard|Oxford|Cambridge|Berkeley|Princeton|Yale)\b',
            r'\b(NASA|NSA|FBI|CIA|WHO|UN|NATO|EU)\b'
        ],
        "location": [
            r'\b(USA|United\s*States|UK|United\s*Kingdom|EU|Europe|Asia|Africa)\b',
            r'\b(New\s*York|London|Paris|Tokyo|Beijing|Berlin|Sydney|Mumbai)\b',
            r'\b(California|Texas|Florida|Bavaria|Ontario|Queensland)\b'
        ],
        "date": [
            r'\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b',
            r'\b(today|yesterday|tomorrow|next\s+\w+|last\s+\w+)\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ],
        "number": [
            r'\b\d+(\.\d+)?\b',
            r'\b(hundred|thousand|million|billion|trillion)\b'
        ]
    }
    
    def __init__(self):
        # Compile patterns
        self.compiled_patterns = {}
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            self.compiled_patterns[entity_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def extract(self, prompt: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities from prompt.
        
        Returns:
            Dict mapping entity type to list of extracted entities
        """
        entities = {}
        
        for entity_type, patterns in self.compiled_patterns.items():
            type_entities = []
            
            for pattern in patterns:
                matches = pattern.finditer(prompt)
                
                for match in matches:
                    value = match.group(0)
                    
                    # Avoid duplicates
                    if not any(e["value"] == value for e in type_entities):
                        type_entities.append({
                            "value": value,
                            "position": match.start(),
                            "length": len(value)
                        })
            
            if type_entities:
                entities[entity_type] = type_entities
        
        return entities
    
    def get_capability_requirements(self, entities: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Determine capability requirements based on extracted entities."""
        capabilities = []
        
        # Code-related entities require code capability
        if "code_language" in entities or "framework" in entities:
            capabilities.append("code")
        
        # Technical domains require reasoning
        if "technology" in entities or "domain" in entities:
            if any("machine learning" in e["value"].lower() or "ai" in e["value"].lower() 
                   for e in entities.get("domain", [])):
                capabilities.append("reasoning")
        
        return list(set(capabilities))  # Remove duplicates


# ============================================================================
# SENTIMENT ANALYZER
# ============================================================================

class SentimentAnalyzer:
    """
    Analyze sentiment of prompts.
    Basic implementation - can be enhanced with ML models.
    """
    
    # Positive and negative word lists
    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "happy", "glad", "pleased", "delighted", "love", "like", "enjoy",
        "perfect", "awesome", "brilliant", "outstanding", "superb",
        "thanks", "thank", "appreciate", "grateful", "helpful", "useful"
    }
    
    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "worst", "poor",
        "sad", "unhappy", "disappointed", "frustrated", "annoyed",
        "hate", "dislike", "useless", "waste", "fail", "failure",
        "problem", "issue", "bug", "error", "crash", "broken",
        "slow", "expensive", "difficult", "confusing", "complex"
    }
    
    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze sentiment of prompt.
        
        Returns:
            Dict with sentiment score (-1 to 1) and label
        """
        prompt_lower = prompt.lower()
        words = re.findall(r'\b[a-z]+\b', prompt_lower)
        
        positive_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        total_count = positive_count + negative_count
        
        if total_count == 0:
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.5
            }
        
        score = (positive_count - negative_count) / total_count
        
        # Determine label
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        # Confidence based on number of sentiment words
        confidence = min(total_count / 10, 0.9) + 0.1
        
        return {
            "score": round(score, 2),
            "label": label,
            "confidence": round(confidence, 2),
            "positive_words": positive_count,
            "negative_words": negative_count
        }


# ============================================================================
# KEYWORD EXTRACTOR
# ============================================================================

class KeywordExtractor:
    """
    Extract important keywords from prompts.
    Uses TF-IDF like approach with stopwords removal.
    """
    
    # Common stopwords to filter out
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "he", "in", "is", "it", "its", "of", "on", "that",
        "the", "to", "was", "were", "will", "with", "would", "could",
        "should", "may", "might", "must", "shall", "this", "these", "those"
    }
    
    def extract(self, prompt: str, max_keywords: int = 10) -> List[Dict[str, Any]]:
        """
        Extract keywords with relevance scores.
        
        Args:
            prompt: Input prompt
            max_keywords: Maximum number of keywords to return
        
        Returns:
            List of keywords with scores
        """
        # Tokenize
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]+\b', prompt.lower())
        
        # Filter stopwords and short words
        words = [w for w in words if w not in self.STOPWORDS and len(w) > 2]
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Calculate scores (normalized frequency)
        max_count = max(word_counts.values()) if word_counts else 1
        
        keywords = []
        for word, count in word_counts.most_common(max_keywords):
            score = count / max_count
            
            # Boost score for capitalized words in original prompt
            if word.title() in prompt or word.upper() in prompt:
                score = min(score + 0.1, 1.0)
            
            keywords.append({
                "keyword": word,
                "count": count,
                "score": round(score, 2)
            })
        
        return keywords


# ============================================================================
# PROMPT ANALYZER
# ============================================================================

class PromptAnalyzer:
    """
    Main prompt analyzer that orchestrates all analysis components.
    
    Provides comprehensive prompt analysis including:
    - Prompt type classification
    - Complexity scoring
    - Language detection
    - Entity extraction
    - Sentiment analysis
    - Keyword extraction
    - Capability requirements
    - Token estimation
    """
    
    def __init__(self):
        self.type_classifier = PromptTypeClassifier()
        self.complexity_scorer = ComplexityScorer()
        self.language_detector = LanguageDetector()
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        
        logger.info("prompt_analyzer_initialized")
    
    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a prompt.
        
        Args:
            prompt: Input prompt
        
        Returns:
            Complete analysis results
        """
        # Trim and normalize
        prompt = prompt.strip()
        
        # Parallel execution of independent analyses
        import asyncio
        
        tasks = [
            self._analyze_type(prompt),
            self._analyze_complexity(prompt),
            self._analyze_language(prompt),
            self._analyze_entities(prompt),
            self._analyze_sentiment(prompt),
            self._analyze_keywords(prompt),
            self._estimate_tokens(prompt)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine results
        analysis = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "length": len(prompt),
            **results[0],  # type
            **results[1],  # complexity
            **results[2],  # language
            **results[3],  # entities
            **results[4],  # sentiment
            **results[5],  # keywords
            **results[6],  # tokens
            "timestamp": self._get_timestamp()
        }
        
        # Derive capability requirements
        analysis["required_capabilities"] = self._derive_capabilities(analysis)
        
        logger.debug(
            "prompt_analyzed",
            prompt_type=analysis["prompt_type"],
            complexity=analysis["complexity_level"],
            language=analysis["language_code"],
            entities=len(analysis.get("entities", {})),
            tokens=analysis.get("estimated_tokens", 0)
        )
        
        return analysis
    
    async def _analyze_type(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt type."""
        scores = self.type_classifier.classify(prompt)
        primary = self.type_classifier.get_primary_type(scores)
        secondary = self.type_classifier.get_secondary_type(scores)
        
        return {
            "prompt_type": primary,
            "prompt_type_secondary": secondary,
            "prompt_type_scores": scores
        }
    
    async def _analyze_complexity(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt complexity."""
        complexity = self.complexity_scorer.calculate(prompt)
        
        return {
            "complexity_score": complexity["score"],
            "complexity_level": complexity["level"],
            "complexity_components": complexity["components"],
            "complexity_metrics": complexity["metrics"]
        }
    
    async def _analyze_language(self, prompt: str) -> Dict[str, Any]:
        """Detect prompt language."""
        language = self.language_detector.detect(prompt)
        
        return {
            "language_code": language["code"],
            "language_name": language["name"],
            "language_confidence": language["confidence"],
            "language_alternatives": language.get("alternatives", [])
        }
    
    async def _analyze_entities(self, prompt: str) -> Dict[str, Any]:
        """Extract entities from prompt."""
        entities = self.entity_extractor.extract(prompt)
        
        return {
            "entities": entities,
            "entity_count": sum(len(e) for e in entities.values())
        }
    
    async def _analyze_sentiment(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt sentiment."""
        sentiment = self.sentiment_analyzer.analyze(prompt)
        
        return {
            "sentiment_score": sentiment["score"],
            "sentiment_label": sentiment["label"],
            "sentiment_confidence": sentiment["confidence"]
        }
    
    async def _analyze_keywords(self, prompt: str) -> Dict[str, Any]:
        """Extract keywords from prompt."""
        keywords = self.keyword_extractor.extract(prompt)
        
        return {
            "keywords": keywords,
            "top_keywords": [k["keyword"] for k in keywords[:5]]
        }
    
    async def _estimate_tokens(self, prompt: str) -> Dict[str, Any]:
        """Estimate token count for prompt."""
        # Simple estimation: ~4 chars per token for English
        char_count = len(prompt)
        estimated_tokens = char_count // 4
        
        return {
            "estimated_tokens": estimated_tokens,
            "estimation_method": "chars_per_token"
        }
    
    def _derive_capabilities(self, analysis: Dict[str, Any]) -> List[str]:
        """Derive required capabilities from analysis."""
        capabilities = set()
        
        # Based on prompt type
        prompt_type = analysis.get("prompt_type", "")
        if prompt_type == "code":
            capabilities.add("code")
        elif prompt_type in ["creative", "conversation"]:
            capabilities.add("chat")
        elif prompt_type == "reasoning":
            capabilities.add("reasoning")
        elif prompt_type == "translation":
            capabilities.add("multilingual")
        
        # Based on entities
        entities = analysis.get("entities", {})
        entity_capabilities = self.entity_extractor.get_capability_requirements(entities)
        capabilities.update(entity_capabilities)
        
        # Based on language
        if analysis.get("language_code") != "en":
            capabilities.add("multilingual")
        
        # Based on complexity
        if analysis.get("complexity_score", 0) > 0.7:
            capabilities.add("reasoning")
        
        return list(capabilities)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_prompt_analyzer = None


def get_prompt_analyzer() -> PromptAnalyzer:
    """Get singleton prompt analyzer instance."""
    global _prompt_analyzer
    if not _prompt_analyzer:
        _prompt_analyzer = PromptAnalyzer()
    return _prompt_analyzer


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PromptAnalyzer",
    "PromptTypeClassifier",
    "ComplexityScorer",
    "LanguageDetector",
    "EntityExtractor",
    "SentimentAnalyzer",
    "KeywordExtractor",
    "get_prompt_analyzer"
]