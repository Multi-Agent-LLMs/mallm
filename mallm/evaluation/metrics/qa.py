import logging
import re
from typing import Any, cast

from evaluate import load

from mallm.evaluation.metrics.metric import Metric

logger = logging.getLogger("mallm")


class MultiChoiceBoolean(Metric):
    """
    A class to evaluate the accuracy on multiple choice/QA tasks.
    """

    _name = "multichoice"
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Final Solution\s*:\s*([A-Z])"
    ANSWER_PATTERN_MULTICHOICE_BACKUP = r"([A-Z])([)\]:]|$)"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:

        reference = reference_texts[0][0]  # first character should always be the label
        match = re.search(MultiChoiceBoolean.ANSWER_PATTERN_MULTICHOICE, generated_text)

        if not match:
            match = re.search(
                MultiChoiceBoolean.ANSWER_PATTERN_MULTICHOICE_BACKUP, generated_text
            )

        if not match:
            logger.warning(f"No pattern match found in answer: {generated_text}")
            return {"correct": None}

        logger.debug(f"Extracted answer: {match.group(1)} from {generated_text}")
        logger.debug(f"Comparing against reference: {reference}")

        extracted_answer = match.group(1)

        score = extracted_answer == reference

        return {"correct": score}


class AnswerabilityBoolean(Metric):
    """
    A class to evaluate the answerability accuracy on QA tasks that include non-answerable questions (i.e., no reference).
    """

    _name = "answerability"

    @staticmethod
    def evaluate(
        generated_text: str, reference_texts: list[str], answer_pattern: str = "unknown"
    ) -> dict[str, Any]:
        if reference_texts == []:
            return {"answerability_correct": answer_pattern in generated_text.lower()}
        return {"answerability_correct": answer_pattern not in generated_text.lower()}


class SquadScore(Metric):
    """
    A class to evaluate the accuracy on squad like tasks that include non-answerable questions (i.e., no reference).
    """

    _name = "squad"
    squad_v2_metric = load("squad_v2")

    @staticmethod
    def evaluate(
        generated_text: str, reference_texts: list[str], answer_pattern: str = "unknown"
    ) -> dict[str, Any]:
        if answer_pattern in generated_text.lower():
            generated_text = ""
        predictions = [
            {"prediction_text": generated_text, "id": "", "no_answer_probability": 0.0}
        ]
        references = [
            {"answers": {"answer_start": [], "text": reference_texts}, "id": ""}
        ]

        scores = cast(
            dict[str, Any],
            SquadScore.squad_v2_metric.compute(
                predictions=predictions, references=references
            ),
        )
        for key in [
            "exact",
            "f1",
            "HasAns_exact",
            "best_exact",
            "best_f1",
            "HasAns_f1",
        ]:
            if key in scores:
                scores[key] /= 100

        return scores
