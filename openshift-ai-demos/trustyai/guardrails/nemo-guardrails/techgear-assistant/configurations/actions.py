from typing import Optional
from nemoguardrails.actions import action

# Constants for configuration
MAX_QUERY_WORDS = 100

# Competitor brands list
COMPETITORS = [
    "apple", "samsung", "dell", "hp", "lenovo", "asus", "acer",
    "microsoft surface", "google pixel", "oneplus", "xiaomi",
    "macbook", "iphone", "galaxy"
]

# Forbidden words list
FORBIDDEN_WORDS = [
    "damn", "hell", "shit", "violence", "murder", "attack", "harm"
]

# Out-of-scope topics
OUT_OF_SCOPE_TOPICS = [
    "medical", "health", "doctor", "medicine", "diagnosis", "treatment",
    "disease", "illness", "hospital", "pharmacy", "therapy", "nurse",
    "legal", "lawyer", "lawsuit", "contract",
    "political", "politics", "government", "president", "election", "vote",
    "investment", "stock", "crypto", "financial", "banking", "credit", "loan"
]

@action(is_system_action=True)
async def check_message_length(context: Optional[dict] = None) -> str:
    """Check if user message exceeds word count limits."""
    user_message = (context or {}).get("user_message", "")
    word_count = len(user_message.split())
    
    if word_count > MAX_QUERY_WORDS:
        return "block_too_long"
    return "allowed"

@action(is_system_action=True)
async def check_forbidden_content(context: Optional[dict] = None) -> str:
    """Check if user message contains forbidden words or inappropriate content."""
    user_message = (context or {}).get("user_message", "").lower()
    
    for word in FORBIDDEN_WORDS:
        if word in user_message:
            return f"blocked_{word}"
    return "allowed"

@action(is_system_action=True)
async def check_competitor_mentions(context: Optional[dict] = None) -> str:
    """Check if user message mentions competitor brands."""
    user_message = (context or {}).get("user_message", "").lower()
    
    for competitor in COMPETITORS:
        if competitor in user_message:
            return f"blocked_competitor_{competitor}"
    return "allowed"

@action(is_system_action=True)
async def check_out_of_scope(context: Optional[dict] = None) -> str:
    """Check if user request is out of scope for e-commerce assistant."""
    user_message = (context or {}).get("user_message", "").lower()
    for topic in OUT_OF_SCOPE_TOPICS:
        if topic in user_message:
            return f"blocked_out_of_scope_{topic}"
    return "allowed"
