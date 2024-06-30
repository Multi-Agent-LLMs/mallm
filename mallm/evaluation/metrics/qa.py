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
    ANSWER_PATTERN_MULTICHOICE_BACKUP = r"([A-E])([)\]:]|$)"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:

        reference = reference_texts[0][0]  # first character should always be the label
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
            logger.debug(
                "No answer pattern detected. Trying to match against freetext reference."
            )
            match = re.search(reference.lower(), generated_text.lower(), flags=re.IGNORECASE)
        else:
            extracted_answer = match.group(1)
            logger.debug(f"Extracted answer: {match.group(1)} from {generated_text}")

        if not match:
            logger.warning(f"No pattern match or text identity found in answer: {generated_text}, reference: {reference}")
            return {"correct": False}

        score = (
            extracted_answer == reference or reference.lower() in generated_text.lower()
        )

        return {"correct": score}


class AnswerabilityBoolean(Metric):
    """
    A class to evaluate the answerability accuracy on QA tasks that include non-answerable questions (i.e., no reference).
    """

    _name = "answerability"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str], answer_pattern: str = "unknown") -> dict[str, Any]:

        if reference_texts == []:
            return {"answerability_correct": answer_pattern in generated_text.lower()}
        return {"answerability_correct": answer_pattern not in generated_text.lower()}
