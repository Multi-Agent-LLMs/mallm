# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of instructions."""
import collections
import json
import random
import re
import string
from collections.abc import Sequence
from typing import Optional, Union

import langdetect
from absl import logging

from mallm.utils.evaluation import instructions_util

_InstructionArgsDtype = Optional[dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
        "My answer is yes.", "My answer is no.", "My answer is maybe.")

# The options of starter keywords.
_STARTER_OPTIONS = ("I would say", "My answer is", "I believe",
                                        "In my opinion", "I think", "I reckon", "I feel",
                                        "From my perspective", "As I see it", "According to me",
                                        "As far as I'm concerned", "To my understanding",
                                        "In my view", "My take on it is", "As per my perception")

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("Any other questions?",
                                     "Is there anything else I can help with?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Section", "SECTION")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500


class Instruction:
    """An instruction template."""

    def __init__(self, instruction_id):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
    """Check the language of the entire response."""

    @classmethod
    def build_description(cls, *, language=None):
        """Build the instruction description.

        Args:
            language: A string representing the expected language of the response. The
                language has to comply to the 97 types defined in
                `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
                ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
                for example, `en` for English, `zh` for Chinese, `fr` for French.

        Returns:
            A string representing the instruction description.
        """
        cls._language = language
        if cls._language is None:
            cls._language = random.choice(list(_LANGUAGES.keys()))
        # TODO(tianjianlu): opens the description generation to more choices.
        cls._description_pattern = (
                f"Your ENTIRE response should be in {language} language, no other " +
                "language is allowed.")
        return cls._description_pattern.format(language=_LANGUAGES[cls._language])

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"language": cls._language}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["language"]

    @classmethod
    def check_following(cls, value):
        """Check if the language of the entire response follows the instruction.

        Args:
            value: A string representing the response.

        Returns:
            True if the language of `value` follows instruction; otherwise False.
        """
        assert isinstance(value, str)

        try:
            return langdetect.detect(value) == cls._language
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                    "Unable to detect language for text %s due to %s", value, e
            )    # refex: disable=pytotw.037
            return True


class NumberOfSentences(Instruction):
    """Check the number of sentences."""

    @classmethod
    def build_description(cls, *, num_sentences=None,
                                                relation=None):
        """Build the instruction description.

        Args:
            num_sentences: An integer specifying the number of sentences as a
                threshold.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.
                Two relational comparisons are supported for now:
                if 'less than', the actual number of sentences < the threshold;
                if 'at least', the actual number of sentences >= the threshold.

        Returns:
            A string representing the instruction description.
        """
        # The number of sentences as a threshold for comparison.
        cls._num_sentences_threshold = num_sentences
        if (cls._num_sentences_threshold is None or
                cls._num_sentences_threshold < 0):
            cls._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

        if relation is None:
            cls._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError("The supported relation for comparison must be in "
                                             f"{_COMPARISON_RELATION}, but {relation} is given.")
        else:
            cls._comparison_relation = relation

        cls._description_pattern = (
                f"Your response should contain {relation} {num_sentences} sentences.")
        return cls._description_pattern.format(
                relation=cls._comparison_relation,
                num_sentences=cls._num_sentences_threshold)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"num_sentences": cls._num_sentences_threshold,
                        "relation": cls._comparison_relation}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "relation"]

    @classmethod
    def check_following(cls, value):
        """Check if the number of sentences follows the instruction.

        Args:
            value: A string representing the response.

        Returns:
            True if the response follows the instruction.

        Raise:
                ValueError if the string in `instruction_args` is not in
                [`less_than`, `at_least`].
        """
        num_sentences = instructions_util.count_sentences(value)
        if cls._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < cls._num_sentences_threshold
        if cls._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= cls._num_sentences_threshold
        return None


class PlaceholderChecker(Instruction):
    """Check the placeholders in template writing."""

    @classmethod
    def build_description(cls, *, num_placeholders=None):
        """Build the instruction description.

        Args:
            num_placeholders: An integer denoting the minimum number of
                placeholders required in the response.

        Returns:
            A string representing the instruction description.
        """
        cls._num_placeholders = num_placeholders
        if cls._num_placeholders is None or cls._num_placeholders < 0:
            cls._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        cls._description_pattern = (
                f"The response must contain at least {num_placeholders} placeholders " +
                "represented by square brackets, such as [address].")
        return cls._description_pattern.format(
                num_placeholders=cls._num_placeholders)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"num_placeholders": cls._num_placeholders}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["num_placeholders"]

    @classmethod
    def check_following(cls, value):
        """Check if the number of placeholders follows the instruction.

        Args:
            value: A string representing the response.

        Returns:
            True if the actual number of placeholders in the response is greater than
            or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= cls._num_placeholders


class BulletListChecker(Instruction):
    """Checks the bullet list in the prompt."""

    @classmethod
    def build_description(cls, *, num_bullets=None):
        """Build the instruction description.

        Args:
            num_bullets: An integer specifying the exact number of bullet lists
                that is required to appear in the response.

        Returns:
            A string representing the instruction description.
        """
        cls._num_bullets = num_bullets
        if cls._num_bullets is None or cls._num_bullets < 0:
            cls._num_bullets = random.randint(1, _NUM_BULLETS)
        cls._description_pattern = (
                f"Your answer must contain exactly {num_bullets} bullet points. " +
                "Use the markdown bullet points such as:\n" +
                "* This is point 1. \n" +
                "* This is point 2")
        return cls._description_pattern.format(
                num_bullets=cls._num_bullets)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"num_bullets": cls._num_bullets}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["num_bullets"]

    @classmethod
    def check_following(cls, value):
        r"""Check if the number of bullet lists meets the requirement.

        Args:
            value: A string representing the response. The response is expected to
                contain some bullet lists that start with `\*`.

        Returns:
            True if the actual number of bullet lists in the response meets the
            requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == cls._num_bullets


class ConstrainedResponseChecker(Instruction):
    """Checks the constrained response."""

    @classmethod
    def build_description(cls):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        cls._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        cls._description_pattern = (
                "Answer with one of the following options: {response_options}")
        return cls._description_pattern.format(
                response_options=cls._constrained_responses)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return []

    @classmethod
    def check_following(cls, value):
        """Checks if the response matches the constrained options.

        Args:
            value: A string representing the response.

        Returns:
            True if the actual response contains one of the options in the constrained
            responses; otherwise False.
        """
        value = value.strip()
        for constrained_response in cls._constrained_responses:
            if constrained_response in value:
                return True
        return False


class ConstrainedStartChecker(Instruction):
    """Checks the response start."""

    @classmethod
    def build_description(cls, *, starter=None):
        """Build the instruction description.

        Args:
            starter: A string representing the keyward that the response should start
                with.

        Returns:
            A string representing the instruction description.
        """
        cls._starter = starter.strip() if isinstance(starter, str) else starter
        if cls._starter is None:
            cls._starter = random.choice(_STARTER_OPTIONS)
        cls._description_pattern = (
                "During the conversation, when it is your turn, " +
                f"please always start with {starter}")
        return cls._description_pattern.format(starter=cls._starter)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"starter": cls._starter}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["starter"]

    @classmethod
    def check_following(cls, value):
        """Checks if the response starts with the constrained keyword or phrase.

        Args:
            value: A string representing the response.

        Returns:
            True if the response starts with the given phrase or keyword that is
            contained in `instruction_args`; otherwise, False.
        """
        response_pattern = r"^\s*" + cls._starter + r".*$"
        response_with_constrained_start = re.search(response_pattern, value,
                                                                                                flags=re.MULTILINE)
        return bool(response_with_constrained_start)


class HighlightSectionChecker(Instruction):
    """Checks the highlighted section."""

    @classmethod
    def build_description(cls, *, num_highlights=None):
        """Build the instruction description.

        Args:
            num_highlights: An integer specifying the minimum number of highlighted
                sections.

        Returns:
            A string representing the instruction description.
        """
        cls._num_highlights = num_highlights
        if cls._num_highlights is None or cls._num_highlights < 0:
            cls._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

        cls._description_pattern = (
                f"Highlight at least {num_highlights} sections in your answer with " +
                "markdown, i.e. *highlighted section*.")

        return cls._description_pattern.format(num_highlights=cls._num_highlights)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"num_highlights": cls._num_highlights}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["num_highlights"]

    @classmethod
    def check_following(cls, value):
        """Checks if the number of highlighted sections meets the requirement.

        Args:
            value: a string repesenting the response. The response is expected to
                contain highlighted sections in the format of *highlighted*.

        Returns:
            True if the actual number of highlighted sections in the format of
            *highlighed sections* meets the minimum requirement; otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1

        return num_highlights >= cls._num_highlights


class SectionChecker(Instruction):
    """Checks the sections."""

    @classmethod
    def build_description(cls, *, section_spliter=None,
                                                num_sections=None):
        """Build the instruction description.

        Args:
            section_spliter: A string represents the section spliter keyword that
                marks a new section, i.e., `Section` or `SECTION`.
            num_sections: An integer specifying the number of sections.

        Returns:
            A string representing the instruction description.
        """
        cls._section_spliter = section_spliter.strip() if isinstance(
                section_spliter, str) else section_spliter
        if cls._section_spliter is None:
            cls._section_spliter = random.choice(_SECTION_SPLITER)

        cls._num_sections = num_sections
        if cls._num_sections is None or cls._num_sections < 0:
            cls._num_sections = random.randint(1, _NUM_SECTIONS)

        cls._description_pattern = (
                f"Your response must have {num_sections} sections. Mark the beginning " +
                f"of each section with {section_spliter} X, such as:\n" +
                f"{section_spliter} 1\n" +
                "[content of section 1]\n" +
                f"{section_spliter} 2\n" +
                "[content of section 2]")

        return cls._description_pattern.format(
                num_sections=cls._num_sections,
                section_spliter=cls._section_spliter)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"section_spliter": cls._section_spliter,
                        "num_sections": cls._num_sections}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["section_spliter", "num_sections"]

    @classmethod
    def check_following(cls, value):
        """Checks the response contains multiple sections.

        Args:
            value: A string representing the response. The response is expected
                to contain multiple sections (number of sections is greater than 1).
                A new section starts with `Section 1`, where the number denotes the
                section index.

        Returns:
            True if the number of sections in the response is greater than or equal to
            the minimum number of sections; otherwise, False.
        """
        section_splitter_patten = r"\s?" + cls._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= cls._num_sections


class ParagraphChecker(Instruction):
    """Checks the paragraphs."""

    @classmethod
    def build_description(cls, *, num_paragraphs=None):
        """Build the instruction description.

        Args:
            num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
            A string representing the instruction description.
        """
        cls._num_paragraphs = num_paragraphs
        if cls._num_paragraphs is None or cls._num_paragraphs < 0:
            cls._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        cls._description_pattern = (
                f"There should be {num_paragraphs} paragraphs. " +
                "Paragraphs are separated with the markdown divider: ***")

        return cls._description_pattern.format(num_paragraphs=cls._num_paragraphs)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"num_paragraphs": cls._num_paragraphs}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs"]

    @classmethod
    def check_following(cls, value):
        """Checks the response contains required number of paragraphs.

        Args:
            value: A string representing the response. The response may contain
                paragraphs that are separated by the markdown divider: `***`.

        Returns:
            True if the actual number of paragraphs is the same as required;
            otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index in {0, len(paragraphs) - 1}:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == cls._num_paragraphs


class PostscriptChecker(Instruction):
    """Checks the postscript."""

    @classmethod
    def build_description(cls, *, postscript_marker=None
                                                ):
        """Build the instruction description.

        Args:
            postscript_marker: A string containing the keyword that marks the start
                of the postscript section.

        Returns:
            A string representing the instruction description.
        """
        cls._postscript_marker = postscript_marker.strip() if isinstance(
                postscript_marker, str) else postscript_marker
        if cls._postscript_marker is None:
            cls._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

        cls._description_pattern = (
                "At the end of your response, please explicitly add a postscript " +
                "starting with {postscript}")

        return cls._description_pattern.format(postscript=cls._postscript_marker)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"postscript_marker": cls._postscript_marker}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["postscript_marker"]

    @classmethod
    def check_following(cls, value):
        """Checks if the response follows the postscript format.

        Args:
            value: a string representing the response. The response is expected to
                contain a postscript section.

        Returns:
            True if the response contains a postscript section starting with
            the keyword containing in the `instruction_args`; otherwise False.
        """
        value = value.lower()
        if cls._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif cls._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + cls._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return bool(postscript)


class RephraseChecker(Instruction):
    """Checks the repharse."""

    @classmethod
    def build_description(cls, *, original_message):
        """Build the instruction description.

        Args:
            original_message: A string representing the original message. The
                rephrased response should only change its words/sentences in between
                its two asterisks, for example, *change me*. Both original and rephrased
                messages should contain the changes in the form of *change me*.

        Returns:
            A string representing the instruction description.
        """
        if not cls.is_change(original_message):
            raise ValueError(f"Message {original_message} does not contain changes "
                                             "in the form of *change me*.")

        cls._reference_without_change = original_message
        cls._description = ("Rephrasing: Your rephrased response should only" +
                                                 "change the words/sentences in between two asterisks" +
                                                 "such as *change me*.")
        return cls._description

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"original_message": cls._reference_without_change}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["original_message"]

    @classmethod
    def check_following(cls, value):
        r"""Checks if the rephrasing follows the instruction.

        Args:
            value: A string representing the response, which is expected to rephras
                the string of `instruction_args`.

        Returns:
            True if `value` and `instruction_args` only differ by the words/sentences
            in between two asterisks such as *change me*; otherwise, False.
        """

        if not cls.is_change(value):
            raise ValueError(f"value {value} does not contain "
                                             "changes in the form of *change me*.")

        response_without_changes = cls.strip_changes(value)
        reference_without_changes = cls.strip_changes(
                cls._reference_without_change)

        return response_without_changes == reference_without_changes

    @classmethod
    def is_change(cls, response):
        """Check if there is change in the response in the form of *change me*."""
        return re.search(r"\*.*\*", response)

    @classmethod
    def strip_changes(cls, response):
        """Strips off the changes."""
        return re.sub(r"\*.*\*", "", response)


class KeywordChecker(Instruction):
    """Check the exisitence of certain keywords."""

    @classmethod
    def build_description(cls, *, keywords=None
                                                ):
        """Build the instruction description.

        Args:
            keywords: A sequence of strings representing the keywords that are
                expected in the response.

        Returns:
            A string representing the instruction description.
        """

        if not keywords:
            cls._keywords = instructions_util.generate_keywords(
                    num_keywords=_NUM_KEYWORDS)
        else:
            cls._keywords = keywords
        cls._keywords = sorted(cls._keywords)

        cls._description_pattern = (f"Include keywords {keywords} in the response.")

        return cls._description_pattern.format(keywords=cls._keywords)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"keywords": cls._keywords}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["keywords"]

    @classmethod
    def check_following(cls, value):
        """Check if the response contain the expected keywords."""
        for keyword in cls._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(Instruction):
    """Check the keyword frequency."""

    @classmethod
    def build_description(cls, *, keyword=None,
                                                frequency=None,
                                                relation=None):
        """Build the instruction description.

        Args:
            keyword: A string representing a keyword that is expected in the response.
            frequency: An integer specifying the number of times `keyword` is expected
                to appear in the response.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.
                Two relational comparisons are supported for now:
                if 'less than', the actual number of occurrences < frequency;
                if 'at least', the actual number of occurrences >= frequency.

        Returns:
            A string representing the instruction description.
        """
        if not keyword:
            cls._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
        else:
            cls._keyword = keyword.strip()

        cls._frequency = frequency
        if cls._frequency is None or cls._frequency < 0:
            cls._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            cls._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError("The supported relation for comparison must be in "
                                             f"{_COMPARISON_RELATION}, but {relation} is given.")
        else:
            cls._comparison_relation = relation

        cls._description_pattern = (
                f"In your response, the word {keyword} should appear {relation} " +
                f"{frequency} times.")

        return cls._description_pattern.format(
                keyword=cls._keyword,
                relation=cls._comparison_relation,
                frequency=cls._frequency)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"keyword": cls._keyword,
                        "frequency": cls._frequency,
                        "relation": cls._comparison_relation}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    @classmethod
    def check_following(cls, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(
                cls._keyword, value, flags=re.IGNORECASE))

        if cls._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < cls._frequency
        if cls._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= cls._frequency
        return None


class NumberOfWords(Instruction):
    """Checks the number of words."""

    @classmethod
    def build_description(cls, *, num_words=None,
                                                relation=None):
        """Build the instruction description.

        Args:
            num_words: An integer specifying the number of words contained in the
                response.
            relation: A string in (`less than`, `at least`), defining the relational
                operator for comparison.
                Two relational comparisons are supported for now:
                if 'less than', the actual number of words < num_words;
                if 'at least', the actual number of words >= num_words.

        Returns:
            A string representing the instruction description.
        """

        cls._num_words = num_words
        if cls._num_words is None or cls._num_words < 0:
            cls._num_words = random.randint(
                    _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
            )

        if relation is None:
            cls._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError("The supported relation for comparison must be in "
                                             f"{_COMPARISON_RELATION}, but {relation} is given.")
        else:
            cls._comparison_relation = relation

        cls._description_pattern = (
                f"Answer with {relation} {num_words} words.")

        return cls._description_pattern.format(
                relation=cls._comparison_relation,
                num_words=cls._num_words)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"num_words": cls._num_words,
                        "relation": cls._comparison_relation}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["num_words", "relation"]

    @classmethod
    def check_following(cls, value):
        """Checks if the response contains the expected number of words."""
        num_words = instructions_util.count_words(value)

        if cls._comparison_relation == _COMPARISON_RELATION[0]:
            return num_words < cls._num_words
        if cls._comparison_relation == _COMPARISON_RELATION[1]:
            return num_words >= cls._num_words
        return None


class JsonFormat(Instruction):
    """Check the Json format."""

    @classmethod
    def build_description(cls):
        cls._description_pattern = (
                "Entire output should be wrapped in JSON format. You can use markdown"
                " ticks such as ```."
        )
        return cls._description_pattern

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return []

    @classmethod
    def check_following(cls, value):
        value = (
                value.strip()
                .removeprefix("```json")
                .removeprefix("```Json")
                .removeprefix("```JSON")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
        )
        try:
            json.loads(value)
        except ValueError as _:
            return False
        return True


class ParagraphFirstWordCheck(Instruction):
    """Check the paragraph and the first word of the nth paragraph."""

    @classmethod
    def build_description(cls, num_paragraphs=None,
                                                nth_paragraph=None,
                                                first_word=None):
        r"""Build the instruction description.

        Args:
            num_paragraphs: An integer indicating the number of paragraphs expected
                in the response. A paragraph is a subset of the string that is
                expected to be separated by '\n\n'.
            nth_paragraph: An integer indicating the paragraph number that we look at.
                Note that n starts from 1.
            first_word: A string that represent the first word of the bth paragraph.

        Returns:
            A string representing the instruction description.
        """
        cls._num_paragraphs = num_paragraphs
        if cls._num_paragraphs is None or cls._num_paragraphs < 0:
            cls._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        cls._nth_paragraph = nth_paragraph
        if (
                cls._nth_paragraph is None
                or cls._nth_paragraph <= 0
                or cls._nth_paragraph > cls._num_paragraphs
        ):
            cls._nth_paragraph = random.randint(1, cls._num_paragraphs + 1)

        cls._first_word = first_word
        if cls._first_word is None:
            cls._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
        cls._first_word = cls._first_word.lower()

        cls._description_pattern = (
                f"There should be {num_paragraphs} paragraphs. " +
                "Paragraphs and only paragraphs are separated with each other by two " +
                "new lines as if it was '\\n\\n' in python. " +
                f"Paragraph {nth_paragraph} must start with word {first_word}.")

        return cls._description_pattern.format(
                num_paragraphs=cls._num_paragraphs,
                nth_paragraph=cls._nth_paragraph,
                first_word=cls._first_word)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"num_paragraphs": cls._num_paragraphs,
                        "nth_paragraph": cls._nth_paragraph,
                        "first_word": cls._first_word}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    @classmethod
    def check_following(cls, value):
        """Checks for required number of paragraphs and correct first word.

        Args:
            value: a string representing the response. The response may contain
                paragraphs that are separated by two new lines and the first word of
                the nth paragraph will have to match a specified word.

        Returns:
            True if the number of paragraphs is the same as required and the first
            word of the specified paragraph is the same as required. Otherwise, false.
        """

        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if cls._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[cls._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        # TODO(jeffrey): make more complex?
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return (
                num_paragraphs == cls._num_paragraphs
                and first_word == cls._first_word
        )


# TODO(jeffrey) add relation - at least/at most?
class KeySentenceChecker(Instruction):
    """Check the existence of certain key sentences."""

    @classmethod
    def build_description(cls, key_sentences=None,
                                                num_sentences=None):
        """Build the instruction description.

        Args:
            key_sentences: A sequences of strings representing the key sentences that
                are expected in the response.
            num_sentences: The number of key sentences that are expected to be seen in
                the response.

        Returns:
            A string representing the instruction description.
        """

        if not key_sentences:
            # TODO(jeffrey) make a generate sentences function? wonderwords package
            cls._key_sentences = {"For now, this is fine."}
        else:
            cls._key_sentences = key_sentences

        if not num_sentences:
            cls._num_sentences = random.randint(1, len(cls._key_sentences))
        else:
            cls._num_sentences = num_sentences

        cls._description_pattern = (
                f"Include {num_sentences} of the following sentences {key_sentences}"
        )

        return cls._description_pattern.format(
                num_sentences=cls._num_sentences, key_sentences=cls._key_sentences
        )

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"num_sentences": cls._num_sentences,
                        "key_sentences": list(cls._key_sentences)}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "key_sentences"]

    @classmethod
    def check_following(cls, value):
        """Checks if the response contains the expected key sentences."""
        count = 0
        sentences = instructions_util.split_into_sentences(value)
        for sentence in cls._key_sentences:
            if sentence in sentences:
                count += 1

        return count == cls._num_sentences


class ForbiddenWords(Instruction):
    """Checks that specified words are not used in response."""

    @classmethod
    def build_description(cls, forbidden_words=None
                                                ):
        """Build the instruction description.

        Args:
            forbidden_words: A sequences of strings respresenting words that are not
                allowed in the response.

        Returns:
            A string representing the instruction description.
        """

        if not forbidden_words:
            cls._forbidden_words = instructions_util.generate_keywords(
                    num_keywords=_NUM_KEYWORDS)
        else:
            cls._forbidden_words = list(set(forbidden_words))
        cls._forbidden_words = sorted(cls._forbidden_words)
        cls._description_pattern = (
                f"Do not include keywords {forbidden_words} in the response."
        )

        return cls._description_pattern.format(
                forbidden_words=cls._forbidden_words
        )

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"forbidden_words": cls._forbidden_words}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["forbidden_words"]

    @classmethod
    def check_following(cls, value):
        """Check if the response does not contain the expected keywords."""
        for word in cls._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


class RephraseParagraph(Instruction):
    """Checks that the paragraph is rephrased."""

    @classmethod
    def build_description(cls, *, original_paragraph, low, high
                                                ):
        """Builds the instruction description.

        Args:
            original_paragraph: A string presenting the original paragraph. The
                rephrases response should have betweeb low-high words in common.
            low: An integer presenting the lower bound of similar words.
            high: An integer representing the upper bound of similar words.

        Returns:
            A string representing the instruction description.
        """
        # TODO(jeffrey) make more encompassing
        cls._original_paragraph = original_paragraph
        cls._low = low
        cls._high = high

        cls._description = ("Rephrase the following paragraph: " +
                                                 f"{original_paragraph}\nYour response should have " +
                                                 f"between {low} and {high} of the same words. " +
                                                 "Words are the same if and only if all of the " +
                                                 "letters, ignoring cases, are the same. For " +
                                                 "example, 'run' is the same as 'Run' but different " +
                                                 "to 'ran'.")

        return cls._description.format(original_paragraph=original_paragraph,
                                                                        low=cls._low, high=cls._high)

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return {"original_paragraph": cls._original_paragraph,
                        "low": cls._low,
                        "high": cls._high}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["original_paragraph", "low", "high"]

    @classmethod
    def check_following(cls, value):
        val_words = re.findall(r"\w+", value.lower())
        original_words = re.findall(r"\w+", cls._original_paragraph.lower())
        similar_words = 0

        dict_val = collections.Counter(val_words)
        dict_original = collections.Counter(original_words)

        for word in dict_original:
            similar_words += min(dict_original[word], dict_val[word])

        return similar_words >= cls._low and similar_words <= cls._high


class TwoResponsesChecker(Instruction):
    """Check that two responses were given."""

    @classmethod
    def build_description(cls):
        """Build the instruction description."""
        cls._description_pattern = (
                "Give two different responses. Responses and only responses should"
                " be separated by 6 asterisk symbols: ******."
        )
        return cls._description_pattern

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyward args of `build_description`."""
        return

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return []

    @classmethod
    def check_following(cls, value):
        """Checks if the response has two different answers.

        Args:
            value: A string representing the response.

        Returns:
            True if two responses are detected and false otherwise.
        """
        valid_responses = []
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index not in {0, len(responses) - 1}:
                    return False
            else:
                valid_responses.append(response)
        return (
                len(valid_responses) == 2
                and valid_responses[0].strip() != valid_responses[1].strip()
        )


class RepeatPromptThenAnswer(Instruction):
    """Checks that Prompt is first repeated then answered."""

    @classmethod
    def build_description(cls, *, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
            prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
            A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        cls._prompt_to_repeat = prompt_to_repeat
        cls._description_pattern = (
                "First repeat the request word for word without change,"
                " then give your answer (1. do not say any words or characters"
                " before repeating the request; 2. the request you need to repeat"
                " does not include this sentence)"
        )
        return cls._description_pattern

    @classmethod
    def get_instruction_args(cls):
        return {"prompt_to_repeat": cls._prompt_to_repeat}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    @classmethod
    def check_following(cls, value):
        return bool(value.strip().lower().startswith(cls._prompt_to_repeat.strip().lower()))


class EndChecker(Instruction):
    """Checks that the prompt ends with a given phrase."""

    @classmethod
    def build_description(cls, *, end_phrase=None):
        """Build the instruction description.

        Args:
            end_phrase: A string representing the phrase the response should end with.

        Returns:
            A string representing the instruction description.
        """
        cls._end_phrase = (
                end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if cls._end_phrase is None:
            cls._end_phrase = random.choice(_ENDING_OPTIONS)
        cls._description_pattern = (
                "Finish your response with this exact phrase {ender}. "
                "No other words should follow this phrase.")
        return cls._description_pattern.format(ender=cls._end_phrase)

    @classmethod
    def get_instruction_args(cls):
        return {"end_phrase": cls._end_phrase}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["end_phrase"]

    @classmethod
    def check_following(cls, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip("\"").lower()
        cls._end_phrase = cls._end_phrase.strip().lower()
        return value.endswith(cls._end_phrase)


class TitleChecker(Instruction):
    """Checks the response for a title."""

    @classmethod
    def build_description(cls):
        """Build the instruction description."""
        cls._description_pattern = (
                "Your answer must contain a title, wrapped in double angular brackets,"
                " such as <<poem of joy>>."
        )
        return cls._description_pattern

    @classmethod
    def get_instruction_args(cls):
        return None

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return []

    @classmethod
    def check_following(cls, value):
        """Checks if the response contains a title."""
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        return any(title.lstrip("<").rstrip(">").strip() for title in titles)


class LetterFrequencyChecker(Instruction):
    """Checks letter frequency."""

    @classmethod
    def build_description(cls, *, letter=None,
                                                let_frequency=None,
                                                let_relation=None):
        """Build the instruction description.

        Args:
            letter: A string representing a letter that is expected in the response.
            let_frequency: An integer specifying the number of times `keyword` is
                expected to appear in the response.
            let_relation: A string in (`less than`, `at least`), defining the
                relational operator for comparison. Two relational comparisons are
                supported for now; if 'less than', the actual number of
                occurrences < frequency; if 'at least', the actual number of
                occurrences >= frequency.

        Returns:
            A string representing the instruction description.
        """
        if (
                not letter
                or len(letter) > 1
                or ord(letter.lower()) < 97
                or ord(letter.lower()) > 122
        ):
            cls._letter = random.choice(list(string.ascii_letters))
        else:
            cls._letter = letter.strip()
        cls._letter = cls._letter.lower()

        cls._frequency = let_frequency
        if cls._frequency is None or cls._frequency < 0:
            cls._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            cls._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                    "The supported relation for comparison must be in "
                    f"{_COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            cls._comparison_relation = let_relation

        cls._description_pattern = (
                f"In your response, the letter {letter} should appear {let_relation}"
                f" {let_frequency} times."
        )

        return cls._description_pattern.format(
                letter=cls._letter,
                let_frequency=cls._frequency,
                let_relation=cls._comparison_relation,
        )

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyword args of build description."""
        return {"letter": cls._letter,
                        "let_frequency": cls._frequency,
                        "let_relation": cls._comparison_relation}

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["letter", "let_frequency", "let_relation"]

    @classmethod
    def check_following(cls, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if cls._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[cls._letter] < cls._frequency
        return letters[cls._letter] >= cls._frequency


class CapitalLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all capital letters."""

    @classmethod
    def build_description(cls):
        """Build the instruction description."""
        cls._description_pattern = (
                "Your entire response should be in English, and in all capital letters."
        )
        return cls._description_pattern

    @classmethod
    def get_instruction_args(cls):
        return None

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return []

    @classmethod
    def check_following(cls, value):
        """Checks that the response is in English and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                    "Unable to detect language for text %s due to %s", value, e
            )    # refex: disable=pytotw.037
            return True


class LowercaseLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all lowercase letters."""

    @classmethod
    def build_description(cls):
        """Build the instruction description."""
        cls._description_pattern = (
                "Your entire response should be in English, and in all lowercase"
                " letters. No capital letters are allowed."
        )
        return cls._description_pattern

    @classmethod
    def get_instruction_args(cls):
        return None

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return []

    @classmethod
    def check_following(cls, value):
        """Checks that the response is in English and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and langdetect.detect(value) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                    "Unable to detect language for text %s due to %s", value, e
            )    # refex: disable=pytotw.037
            return True


class CommaChecker(Instruction):
    """Checks the response for no commas."""

    @classmethod
    def build_description(cls):
        """Build the instruction description."""
        cls._description_pattern = (
                "In your entire response, refrain from the use of any commas."
        )
        return cls._description_pattern

    @classmethod
    def get_instruction_args(cls):
        return None

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return []

    @classmethod
    def check_following(cls, value):
        """Checks that the response does not contain commas."""
        return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
    """Checks frequency of words with all capital letters."""

    @classmethod
    def build_description(
            cls,
            capital_frequency=None,
            capital_relation=None,
    ):
        """Build the instruction description.

        Args:
            capital_frequency: An integer that represents the number of words that
                should be in all capital letters.
            capital_relation: A string that is 'at least' or 'at most' that refers to
                the frequency.

        Returns:
            A string representing the instruction description.
        """
        cls._frequency = capital_frequency
        if cls._frequency is None:
            cls._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

        cls._comparison_relation = capital_relation
        if capital_relation is None:
            cls._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                    "The supported relation for comparison must be in "
                    f"{_COMPARISON_RELATION}, but {capital_relation} is given."
            )

        cls._description_pattern = (
                "In your response, words with all capital letters should appear"
                " {relation} {frequency} times."
        )

        return cls._description_pattern.format(
                frequency=cls._frequency, relation=cls._comparison_relation
        )

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyword args of build description."""
        return {
                "capital_frequency": cls._frequency,
                "capital_relation": cls._comparison_relation,
        }

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return ["capital_frequency", "capital_relation"]

    @classmethod
    def check_following(cls, value):
        """Checks the frequency of words with all capital letters."""
        # Hyphenated words will count as one word
        words = instructions_util.nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words = len(capital_words)

        if cls._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < cls._frequency
        return capital_words >= cls._frequency


class QuotationChecker(Instruction):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
                "Wrap your entire response with double quotation marks."
        )
        return self._description_pattern

    @classmethod
    def get_instruction_args(cls):
        """Returns the keyword args of build description."""
        return

    @classmethod
    def get_instruction_args_keys(cls):
        """Returns the args keys of `build_description`."""
        return []

    @classmethod
    def check_following(cls, value):
        """Checks if the response is wrapped with double quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'
