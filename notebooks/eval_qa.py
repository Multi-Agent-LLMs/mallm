from collections import defaultdict
import json
import sys
from dataclasses import dataclass
from typing import List, Optional
from scipy.stats import norm


@dataclass
class PersonaAttributes:
    role: str
    extraversion: str
    agreeableness: str
    conscientiousness: str
    neuroticism: str
    openness: str
    experience: str
    gender: str


@dataclass
class Persona:
    agentId: str
    persona: str
    personaDescription: str
    personaAttributes: PersonaAttributes


@dataclass
class Scores:
    correct: bool


@dataclass
class QAData:
    dataset: str
    exampleId: str
    datasetId: str
    instruction: str
    coordinatorId: str
    paradigm: str
    personas: List[Persona]
    input: List[str]
    context: Optional[None]
    answer: str
    references: List[str]
    decision_success: bool
    turns: int
    clockSeconds: float
    scores: Scores


def create_persona_attributes(attributes_dict):
    """Safely create PersonaAttributes from a dictionary."""
    required_keys = [
        "role",
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "neuroticism",
        "openness",
        "experience",
        "gender",
    ]
    if "role" not in attributes_dict:
        attributes_dict["role"] = "Neutral"
    if all(key in attributes_dict for key in required_keys):
        return PersonaAttributes(**attributes_dict)
    else:
        missing_keys = [key for key in required_keys if key not in attributes_dict]
        raise ValueError(f"Missing keys in persona attributes: {missing_keys}")


def parse_dataset_file(filename):
    """Load a list of Dataset objects from a JSON file."""
    with open(filename, "r") as file:
        data = json.load(file)
        datasets = [
            QAData(
                dataset=item["dataset"],
                exampleId=item["exampleId"],
                datasetId=item["datasetId"],
                instruction=item["instruction"],
                coordinatorId=item["coordinatorId"],
                paradigm=item["paradigm"],
                personas=[
                    Persona(
                        agentId=persona["agentId"],
                        persona=persona["persona"],
                        personaDescription=persona["personaDescription"],
                        personaAttributes=create_persona_attributes(
                            persona["personaAttributes"]
                        ),
                    )
                    for persona in item["personas"]
                ],
                input=item["input"],
                context=item["context"],
                answer=item["answer"],
                references=item["references"],
                decision_success=item["decision_success"],
                turns=item["turns"],
                clockSeconds=item["clockSeconds"],
                scores=Scores(**item["scores"]),
            )
            for item in data
        ]
    return datasets


def two_proportion_z_test(successes1, n1, successes2, n2):
    """Calculate the z-score and p-value for a two-proportion z-test."""
    p1 = successes1 / n1
    p2 = successes2 / n2
    p = (successes1 + successes2) / (n1 + n2)
    z = (p1 - p2) / ((p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5)
    p_value = 2 * norm.sf(abs(z))
    return p_value


if __name__ == "__main__":
    filename = sys.argv[1]
    qaData = parse_dataset_file(filename)
    print(f"Loaded {len(qaData)} datasets.")

    # Collect all possible values for each attribute
    attribute_values: dict[str, set] = defaultdict(set)
    for run in qaData:
        for agent in run.personas:
            persona = agent.personaAttributes
            # attribute_values['role'].add(persona.role)
            attribute_values["extraversion"].add(persona.extraversion)
            attribute_values["agreeableness"].add(persona.agreeableness)
            attribute_values["conscientiousness"].add(persona.conscientiousness)
            attribute_values["neuroticism"].add(persona.neuroticism)
            attribute_values["openness"].add(persona.openness)
            attribute_values["experience"].add(persona.experience)
            attribute_values["gender"].add(persona.gender)

    # remove "neutral" case-sensitive
    for key in attribute_values.keys():
        if "neutral" in attribute_values[key]:
            attribute_values[key].remove("neutral")

    # Calculate and pretty print results for each attribute and its values
    for attribute, values in attribute_values.items():
        for value in values:
            fdata = [
                run
                for run in qaData
                if all(
                    getattr(agent.personaAttributes, attribute) == value
                    for agent in run.personas
                )
            ]
            if fdata:
                oppvalue = "high" if value == "low" else "low"
                notfdata = [
                    run
                    for run in qaData
                    if all(
                        getattr(agent.personaAttributes, attribute) == oppvalue
                        for agent in run.personas
                    )
                ]

                pvalue = (
                    two_proportion_z_test(
                        sum(run.scores.correct for run in fdata),
                        len(fdata),
                        sum(run.scores.correct for run in notfdata),
                        len(notfdata),
                    )
                    if notfdata
                    else 1
                )

                average_score = sum(run.scores.correct for run in fdata) / len(fdata)
                print(
                    f"{attribute} = {value}: {average_score:.1%} (n = {len(fdata)}) (p = {pvalue:.3f})"
                )

    personality_results = {}
    # add all "high"/"low" personality combinations
    import itertools

    for personality in itertools.product(["high", "low"], repeat=5):
        personality_results[personality] = {"correct": 0, "total": 0}

    for run in qaData:
        for persona in run.personas:
            attributes = persona.personaAttributes
            personality = (
                attributes.extraversion,
                attributes.agreeableness,
                attributes.conscientiousness,
                attributes.neuroticism,
                attributes.openness,
            )
            if personality not in personality_results:
                personality_results[personality] = {"correct": 0, "total": 0}
            personality_results[personality]["total"] += 1
            personality_results[personality]["correct"] += run.scores.correct

    # sort
    sorted_personality_results = sorted(
        personality_results.items(),
        key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0,
        reverse=True,
    )

    # pretty print
    for personality, results in sorted_personality_results:
        correct = results["correct"]
        total = results["total"]
        accuracy = correct / total if total > 0 else 0
        # convert 5-tuple to string with uppercase/lowercase first letter of each personality trait
        personality = "".join(
            f"{name[0].upper() if trait == "high" else name[0].lower()}" for trait, name in zip(personality, ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"])
        )
        print(f"{personality}: {accuracy:.1%} (n = {total})")
