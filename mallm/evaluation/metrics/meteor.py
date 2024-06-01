from typing import Any

from nltk import download, word_tokenize
from nltk.translate.meteor_score import meteor_score

from mallm.evaluation.metrics.metric import Metric

download("wordnet")


class METEOR(Metric):
    """
    A class to evaluate the METEOR score for text generation tasks.
    """

    _name = "METEOR"

    @staticmethod
    def evaluate(generated_text: str, reference_texts: list[str]) -> dict[str, Any]:
        # Tokenize the input texts
        generated_tokens = word_tokenize(generated_text)
        reference_tokens = word_tokenize(reference_texts[0])
        # Calculate METEOR score
        score = meteor_score([generated_tokens], reference_tokens)
        return {"meteor": score}
