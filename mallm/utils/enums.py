from enum import Enum


class DecisionAlteration(Enum):
    PUBLIC = "public"
    FACTS = "facts"
    HISTORY = "history"
    CONFIDENCE = "confidence"
    CONFIDENCE_LOG_PROBS = "confidence_log_probs"
    CONFIDENCE_PROMPTED = "confidence_prompted"
    CONFIDENCE_CONSISTENCY = "confidence_consistency"
    ANONYMOUS = "anonymous"
