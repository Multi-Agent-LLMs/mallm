from typing import Any

from datasets import load_metric
from nltk import word_tokenize

from .metric import Metric

# from nltk.translate.bleu_score import sentence_bleu    # preferred but broken with python 3.12 at commit time


class BLEU(Metric):
    """
    A class to evaluate the BLEU score for text generation tasks.
    """

    _name = "BLEU"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:
        # Tokenize the input texts
        generated_tokens = [word_tokenize(generated_text)]
        reference_tokens = [[word_tokenize(r)] for r in reference_texts]
        # Calculate BLEU score
        score = load_metric("bleu", trust_remote_code=True).compute(
            references=reference_tokens, predictions=generated_tokens
        )
        return {"bleu": score["bleu"]}
