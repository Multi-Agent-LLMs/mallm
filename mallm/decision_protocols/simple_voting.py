import logging
import re
from collections import Counter
from typing import Any, Optional

from mallm.agents.panelist import Panelist
from mallm.decision_protocols.protocol import DecisionProtocol
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.config import Config
from mallm.utils.enums import DecisionAlteration
from mallm.utils.types import Agreement, VotingResult, VotingResultList, WorkerFunctions

logger = logging.getLogger("mallm")


def _parse_trailing_digit_block(
    vote_str: str, num_answers: int, in_range
) -> Optional[int]:
    """
    Last contiguous digit run at the end of the string (after trailing spaces).
    If that integer is out of range and ``num_answers <= 10``, use the last digit
    of the run when it alone is a valid index (e.g. ``...12`` -> ``2``).
    """
    s = vote_str.rstrip()
    if not s:
        return None
    i = len(s) - 1
    while i >= 0 and not s[i].isdigit():
        i -= 1
    if i < 0:
        return None
    j = i
    while j >= 0 and s[j].isdigit():
        j -= 1
    block = s[j + 1 : i + 1]
    v = int(block)
    if in_range(v):
        return v
    if num_answers <= 10 and len(block) > 1:
        last_d = int(block[-1])
        if in_range(last_d):
            return last_d
    return None


def parse_vote_from_last_line(vote_str: str, num_answers: int) -> Optional[int]:
    """
    Extract a solution index from an LLM voting reply. Tiers run from strict to
    permissive so obvious formats win; loose tiers reduce invalid-vote retries.

    1. Bottom-up: a line that is **only** digits and in ``[0, num_answers)``.
    2. Bottom-up: ``Final Answer: <int>`` (case-insensitive).
    3. Bottom-up: common phrases (``Vote: N``, ``vote for solution N``, etc.).
    4. **Rightmost** ``\\d+`` substring in the full text whose value is in range.
    5. Trailing digit block at end of string (plus last-digit fallback for small
       ``num_answers``; see ``_parse_trailing_digit_block``).
    """
    if num_answers <= 0:
        return None
    if not isinstance(vote_str, str):
        vote_str = str(vote_str)

    def in_range(v: int) -> bool:
        return 0 <= v < num_answers

    lines = [ln.strip() for ln in vote_str.splitlines() if ln.strip()]

    for line in reversed(lines):
        if re.fullmatch(r"\d+", line):
            v = int(line)
            if in_range(v):
                return v

    final_answer_re = re.compile(r"(?i)final\s+answer\s*:\s*(\d+)")
    for line in reversed(lines):
        m = final_answer_re.search(line)
        if m:
            v = int(m.group(1))
            if in_range(v):
                return v

    # Phrases often used when the model ignores "answer with the number only".
    line_phrase_res = (
        re.compile(r"(?i)\b(?:my\s+)?vote\s*:\s*(\d+)\b"),
        re.compile(r"(?i)\bcast(?:ing)?\s+(?:my\s+)?vote\s+for\s+solution\s*(\d+)\b"),
        re.compile(r"(?i)\bvote\s+for\s+(?:solution|option)\s*#?\s*(\d+)\b"),
        re.compile(r"(?i)\bvote\s+for\s+\*\*solution\s*(\d+)\*\*"),
        re.compile(r"(?i)\b(?:choose|pick|select)(?:ing)?\s+solution\s*(\d+)\b"),
        re.compile(r"(?i)\b(?:solution|option)\s*#?\s*(\d+)\s*(?:\.|!|\)|\]|,|;|:)?\s*$"),
        re.compile(r"(?i)\*\*solution\s*(\d+)\*\*"),
    )
    for line in reversed(lines):
        for rx in line_phrase_res:
            m = rx.search(line)
            if m:
                v = int(m.group(1))
                if in_range(v):
                    return v

    for m in reversed(list(re.finditer(r"\d+", vote_str))):
        v = int(m.group(0))
        if in_range(v):
            return v

    tail = _parse_trailing_digit_block(vote_str, num_answers, in_range)
    if tail is not None:
        return tail

    return None


class SimpleVoting(DecisionProtocol):
    """
    The Voting decision protocol allows panelists to vote for the best answer after a certain number of turns.
    """

    _name = "simple_voting"

    def __init__(
        self,
        panelists: list[Panelist],
        num_neutral_agents: int,
        worker_functions: WorkerFunctions,
        vote_turn: int = 3,
    ) -> None:
        super().__init__(panelists, num_neutral_agents, worker_functions)
        self.vote_turn = vote_turn

    def make_decision(
        self,
        agreements: list[Agreement],
        turn: int,
        agent_index: int,
        task: str,
        question: str,
        config: Config,
    ) -> tuple[str, bool, list[Agreement], str, Optional[VotingResultList]]:
        if len(agreements) > self.total_agents:
            agreements = agreements[-self.total_agents :]
        if turn < self.vote_turn or agent_index != self.total_agents - 1:
            return "", False, agreements, "", None

        final_answers_with_confidence, voting_process_string = (
            self.generate_final_answers(agreements, question, task)
        )

        decision, final_answer, results, voting_process_string = (
            self.vote_with_alterations(
                final_answers_with_confidence,
                question,
                task,
                voting_process_string,
                self._name,
                ResponseGenerator.generate_voting_prompt,
                config.voting_protocols_with_alterations,
            )
        )
        return (
            final_answer,
            decision,
            agreements,
            voting_process_string,
            results,
        )

    def process_results(
        self,
        all_votes: dict[str, VotingResult],
        alteration: DecisionAlteration,
        final_answers: list[str],
        votes: list[int],
    ) -> dict[str, VotingResult]:
        winners = []
        most_voted = -1
        if votes:
            vote_counts = Counter(votes)
            most_common = vote_counts.most_common()
            most_voted = most_common[0][0]
            most_voted_count = most_common[0][1]

            # Check if there are multiple winners with the same vote count
            winners = [
                candidate
                for candidate, count in vote_counts.items()
                if count == most_voted_count
            ]

        if len(winners) == 1:
            all_votes[alteration.value] = VotingResult(
                votes=votes,
                most_voted=most_voted,
                final_answer=final_answers[most_voted],
                agreed=True,
            )
            logger.info(
                f"Voted for answer from agent {self.panelists[most_voted].persona}"
            )
        else:
            all_votes[alteration.value] = VotingResult(
                votes=votes,
                most_voted=-1,
                final_answer="",
                agreed=False,
            )
            logger.info("There was a tie. Going for another round of voting.")
        return all_votes

    def process_votes(
        self,
        final_answers: list[str],
        panelist: Panelist,
        vote_str: str,
        vote: Any,
        voting_process_string: str,
    ) -> tuple[str, Any, bool, str]:
        success = False
        vote_int = parse_vote_from_last_line(vote_str, len(final_answers))
        if vote_int is not None:
            vote.append(vote_int)
            logger.info(
                f"{panelist.persona} voted for answer from {self.panelists[vote_int].persona}"
            )
            voting_process_string += f"{panelist.persona} voted for answer from {self.panelists[vote_int].persona}\n"
            success = True
        return vote_str, vote, success, voting_process_string
