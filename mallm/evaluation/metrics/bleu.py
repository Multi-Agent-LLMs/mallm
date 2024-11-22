from typing import Any, Optional

from evaluate import load

from mallm.evaluation.metrics.metric import Metric

# from nltk.translate.bleu_score import sentence_bleu    # TODO: preferred but broken with python 3.12 at commit time


class BLEU(Metric):
    """
    A class to evaluate the BLEU score for text generation tasks.
    """

    _name = "BLEU"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str], dataset_id: Optional[str]) -> dict[str, Any]:
        # Calculate BLEU score
        score = load("bleu", trust_remote_code=True).compute(
            references=[reference_texts], predictions=[generated_text]
        )
        return {"bleu": score["bleu"]}
