import logging
import re
from typing import Any

from mallm.evaluation.metrics.metric import Metric

logger = logging.getLogger("mallm")


class MultiChoiceBoolean(Metric):
    """
    A class to evaluate the accuracy on multiple choice/QA tasks.
    """

    _name = "multichoice"
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Final Solution\s*:\s*([A-E])"
    ANSWER_PATTERN_MULTICHOICE_BACKUP = r"([A-E])(\W|$)"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:

        reference = reference_texts[0]
        match = re.search(MultiChoiceBoolean.ANSWER_PATTERN_MULTICHOICE, generated_text)

        if not match:
            match = re.search(
                MultiChoiceBoolean.ANSWER_PATTERN_MULTICHOICE_BACKUP, generated_text
            )

        if not match:
            logger.warning(f"No match found in answer: {generated_text}")
            return {"score": None}

        logger.debug(f"Extracted answer: {match.group(1)} from {generated_text}")

        extracted_answer = match.group(1)

        assert len(reference) == 1, "Reference should be a single character."
        score = extracted_answer == reference

        return {"correct": score}
