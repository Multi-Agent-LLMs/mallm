import json
import random

from experiments.data.data_download import DatasetDownloader


class GPQADownloader(DatasetDownloader):
    def custom_download(self):
        pass

    def __init__(self):
        super().__init__(
            name="gpqa", dataset_name="Idavidrein/gpqa", version="gpqa_extended"
        )

    def process_data(self):
        data = self.shuffle_and_select()
        examples = []

        for sample in data.iter(batch_size=1):
            answers = self._format_answers(sample)
            correct_answer_index = answers.index(
                json.dumps(sample["Correct Answer"][0])
            )
            correct_answer_label = chr(65 + correct_answer_index)

            question_text = self._clean_text(sample["Question"][0])
            formatted_answers = self._format_answer_choices(answers)
            question_and_answers = f"{question_text}\n\n" + "\n".join(formatted_answers)

            example = {
                "exampleId": sample["Record ID"][0],
                "datasetId": sample["Canary String"][0],
                "input": question_and_answers,
                "context": None,
                "references": [correct_answer_label],
                "personas": None,
            }
            examples.append(json.dumps(example) + "\n")

        self.save_to_json("".join(examples))

    def _format_answers(self, sample):
        answers = [json.dumps(sample[f"Incorrect Answer {i}"][0]) for i in range(1, 4)]
        answers.insert(0, json.dumps(sample["Correct Answer"][0]))
        random.shuffle(answers)
        return answers

    def _format_answer_choices(self, answers):
        return [f" {chr(65 + i)}. {self._clean_text(answers[i])}" for i in range(4)]

    def _clean_text(self, text):
        return (
            text.replace("\n", " ")
            .replace("\r", " ")
            .replace('"', "")
            .replace("\\n", " ")
        )
