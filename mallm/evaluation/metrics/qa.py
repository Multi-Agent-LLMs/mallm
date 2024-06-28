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
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Final Solution\s*:\s*([A-D])"
    ANSWER_PATTERN_MULTICHOICE_BACKUP = r"([A-D])(\W|$)"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:

        reference = reference_texts[0]
        match = re.search(MultiChoiceBoolean.ANSWER_PATTERN_MULTICHOICE, generated_text)
        extracted_answer = None

        if not match:
            match = re.search(
                MultiChoiceBoolean.ANSWER_PATTERN_MULTICHOICE_BACKUP, generated_text
            )

        if not match:
            reference = re.sub(
                MultiChoiceBoolean.ANSWER_PATTERN_MULTICHOICE, "", reference
            ).strip()
            reference = re.sub(
                MultiChoiceBoolean.ANSWER_PATTERN_MULTICHOICE_BACKUP, "", reference
            ).strip()
            match = re.search(reference, generated_text, flags=re.IGNORECASE)
            logger.debug(
                "No answer pattern detected. Trying to match against freetext reference."
            )
        else:
            extracted_answer = match.group(1)
            logger.debug(f"Extracted answer: {match.group(1)} from {generated_text}")

        if not match:
            logger.warning(f"No match found in answer: {generated_text}")
            return {"correct": False}

        score = (
            extracted_answer
            == reference  # or reference.lower() in generated_text.lower()
        )

        return {"correct": score}
