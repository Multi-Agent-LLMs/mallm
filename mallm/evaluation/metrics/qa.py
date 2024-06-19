import logging
import re
from typing import Any

from mallm.evaluation.metrics.metric import Metric

logger = logging.getLogger("mallm")

class QA(Metric):
    """
    A class to evaluate the accuracy on multiple choice/QA tasks.
    """

    _name = "qa"
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
    ANSWER_PATTERN_MULTICHOICE_BACKUP = r"([A-D])(\W|$)"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:

        reference = reference_texts[0]
        match = re.search(QA.ANSWER_PATTERN_MULTICHOICE, generated_text)

        if not match:
            match = re.search(QA.ANSWER_PATTERN_MULTICHOICE_BACKUP, generated_text)

        if not match:
            logger.warning(f"No match found in answer: {generated_text}")
            return {"score": None}
        
        logger.debug(f"Extracted answer: {match.group(1)} from {generated_text}")

        extracted_answer = match.group(1)

        assert len(reference) == 1, "Reference should be a single character."
        score = extracted_answer == reference

        return {"correct": score}
