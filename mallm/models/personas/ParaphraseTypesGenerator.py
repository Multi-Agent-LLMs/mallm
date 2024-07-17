import logging

from mallm.models.Chat import Chat
from mallm.models.personas.PersonaGenerator import PersonaGenerator
from mallm.utils.types import InputExample

logger = logging.getLogger("mallm")


class ParaphraseTypesGenerator(PersonaGenerator):
    def __init__(self, llm: Chat):
        self.llm = llm
        # definitions of paraphrase types started from https://github.com/worta/generate_apt_paraphrases/blob/main/definitions.json and expanded with ChatGPT on 2024/07/16
        self.paraphrase_types = (
            {
                "inflectional changes": "Inflectional changes consist of changing inflectional affixes of words.",
                "modal verb changes": "Modal verb changes consist of replacing one modal verb with another, altering the modality of the sentence but preserving its overall meaning.",
                "derivational changes": "Derivational Changes consist of changes of category with or without using derivational affixes. These changes imply a syntactic change in the sentence in which they occur.",
                "spelling changes": "Spelling and format changes comprise changes in the spelling and format of lexical (or functional) units, such as case changes, abbreviations, or digit/letter alternations.",
                "same polarity substitution (habitual)": "Same polarity substitution (habitual) involves replacing a word or phrase with another that has a similar habitual or general meaning, maintaining the original sentence's polarity.",
                "same polarity substitution (contextual)": "Same Polarity Substitution consists of changing one lexical (or functional) unit for another with approximately the same meaning. Among the linguistic mechanisms of this type, we find synonymy, general/specific substitutions, or exact/approximate alternations.",
                "same polarity substitution (named entity)": "Same polarity substitution (named entity) involves substituting one named entity with another, while keeping the overall sentence meaning and polarity intact.",
                "change of format": "Change of format refers to alterations in the format of the text, such as switching between numerical and written forms, or changing date formats.",
                "opposite polarity substitution (habitual)": "Opposite polarity substitution (habitual) consists of replacing a word or phrase with another that has an opposite habitual meaning, altering the sentence's overall polarity.",
                "opposite polarity substitution (contextual)": "Opposite polarity substitution (contextual) involves substituting a word or phrase with another that has an opposite meaning in a specific context, changing the sentence's polarity.",
                "synthetic/analytic substitution": "Synthetic/analytic substitution consists of changing synthetic structures for analytic structures, and vice versa. This type comprises mechanisms such as compounding/ decomposition, light element, or lexically emptied specifier additions/deletions, or alternations affecting genitives and possessives.",
                "converse substitution": "Converse substitution involves replacing a word or phrase with its converse, altering the perspective but preserving the overall relationship described in the sentence.",
                "diathesis alternation": "Diathesis alternation consists of changing the voice of a verb in a sentence, such as switching from active to passive voice, or vice versa.",
                "negation switching": "Negation switching involves introducing or removing negation from a sentence, altering its affirmative or negative status.",
                "ellipsis": "Ellipsis consists of omitting parts of a sentence that are understood from context, creating a more concise expression without losing meaning.",
                "coordination changes": "Coordination changes involve altering the structure of coordinated elements in a sentence, such as changing conjunctions or the order of coordinated phrases.",
                "subordination and nesting changes": "Subordination and nesting changes consist of changes in which one of the members of the pair contains a subordination or nested element, which is not present, or changes its position and/or form within the other member of the pair.",
                "punctuation changes": "Punctuation and format changes consist of any change in the punctuation or format of a sentence (not of a lexical unit, cf. lexicon-based changes).",
                "direct/indirect style alternations": "Direct/indirect style alternations involve changing the sentence from direct speech to indirect speech or vice versa, altering the reporting style.",
                "sentence modality changes": "Sentence modality changes involve altering the modality of a sentence, such as changing from declarative to interrogative, or from imperative to subjunctive.",
                "syntax/discourse structure changes": "Syntax/discourse structure changes involve alterations in the syntactic structure or the overall discourse organization of the text.",
                "addition/deletion": "Addition/Deletion consists of all additions/deletions of lexical and functional units.",
                "change of order": "Change of order includes any type of change of order from the word level to the sentence level.",
                "semantic based": "Semantics-based changes are those that involve a different lexicalization of the same content units. These changes affect more than one lexical unit and a clear-cut division of these units in the mapping between the two members of the paraphrase pair is not possible."
            }
        )

    def generate_personas(
        self, task_description: str, num_agents: int, sample: InputExample
    ) -> list[dict[str, str]]:
        if not sample.context:
            logger.error(
                "Failed to generate personas because there were no paraphrase types provided in the context."
            )
            raise Exception("Failed to generate personas.")

        context = sample.context[0].replace("Paraphrase Types: ", "")
        paraphrase_types_list = [ptype.strip().lower() for ptype in context.split(",") if ptype.strip().lower() in self.paraphrase_types.keys()]   # excludes extremes identity, non-paraphrase, and entailment
        agents = [
            {"role": "You are an expert in " + p + ". Make sure your paraphrase type is properly incorporated into the solution.", "description": self.paraphrase_types[p]}
            for p in paraphrase_types_list
        ]

        return agents
