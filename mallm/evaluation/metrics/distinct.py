from typing import Any

from distinct_n import distinct_n_sentence_level

from mallm.evaluation.metrics.metric import Metric


class Distinct(Metric):
    """
    A class to evaluate generated texts by Distinct-N score based on Jiwei Li et. al. (https://arxiv.org/pdf/1510.03055)
    """

    _name = "Distinct"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:
        score1 = distinct_n_sentence_level(generated_text, 1)
        score2 = distinct_n_sentence_level(generated_text, 2)
        return {"distinct1": score1, "distinct2": score2}
