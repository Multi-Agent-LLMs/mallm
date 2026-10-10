"""Tests for simple voting: last-line vote parsing."""

from mallm.agents.panelist import Panelist
from mallm.coordinator import Coordinator
from mallm.decision_protocols.simple_voting import SimpleVoting, parse_vote_from_last_line
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator

coordinator = Coordinator(None, None)
response_generator = FreeTextResponseGenerator(None)
panelists = [
    Panelist(None, None, coordinator, response_generator, f"P{i}", "")
    for i in range(3)
]


def test_parse_last_line_three_answers_plain_digit():
    assert parse_vote_from_last_line("1", 3) == 1
    assert parse_vote_from_last_line("  2  ", 3) == 2
    assert parse_vote_from_last_line("0", 3) == 0


def test_parse_last_line_three_answers_ignores_earlier_numbers():
    """Regression: do not concatenate digits from explanation text."""
    text = (
        "I compared option 1 and option 2.\n"
        "The first solution mentions 2024.\n"
        "2"
    )
    assert parse_vote_from_last_line(text, 3) == 2


def test_parse_final_answer_line_with_trailing_tags():
    """GLM-style last line: not a bare digit, but Final Answer: N before tags."""
    text = (
        "Therefore, I vote for **Solution 1**.\n\n"
        "Final Answer: 1</think>1. Asking to re-vote."
    )
    assert parse_vote_from_last_line(text, 3) == 1


def test_parse_vote_colon_and_trailing_punct():
    assert parse_vote_from_last_line("vote: 1", 3) == 1
    assert parse_vote_from_last_line("My vote: 2", 3) == 2
    assert parse_vote_from_last_line("1.", 3) == 1


def test_parse_markdown_solution_phrase():
    text = "After review, I vote for **Solution 1** because it is clearest."
    assert parse_vote_from_last_line(text, 3) == 1


def test_parse_rightmost_in_range_token():
    """Prefer the rightmost digit-group that is a valid index (not left years)."""
    text = "We compared scores from 2024 and earlier drafts; the best is 2"
    assert parse_vote_from_last_line(text, 3) == 2


def test_parse_trailing_digit_block_with_last_digit_fallback():
    """Multi-digit at end: last digit used as vote when whole number is out of range."""
    assert parse_vote_from_last_line("bogus 12", 3) == 2


def test_parse_last_line_rejects_out_of_range():
    assert parse_vote_from_last_line("3", 3) is None
    assert parse_vote_from_last_line("9", 3) is None


def test_parse_empty():
    assert parse_vote_from_last_line("", 3) is None
    assert parse_vote_from_last_line("\n\n", 3) is None


def test_parse_eleven_answers_uses_integer_line():
    assert parse_vote_from_last_line("10", 11) == 10
    assert parse_vote_from_last_line("  5  ", 11) == 5
    assert parse_vote_from_last_line("11", 11) is None


def test_process_votes_end_to_end():
    sv = SimpleVoting(panelists, num_neutral_agents=0, worker_functions=None)  # type: ignore[arg-type]
    votes: list[int] = []
    _, votes, ok, _ = sv.process_votes(
        ["a", "b", "c"],
        panelists[0],
        "Long reasoning about 1 and 2.\n\n1\n",
        votes,
        "",
    )
    assert ok is True
    assert votes == [1]
