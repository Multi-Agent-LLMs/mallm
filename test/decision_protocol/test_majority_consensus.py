from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.coordinator import Coordinator
from mallm.decision_protocol.majority import HybridMajorityConsensus, UnanimityConsensus
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.utils.types import Agreement

coordinator = Coordinator(None, None)
response_generator = FreeTextResponseGenerator(None)
panelists = [
    Panelist(None, None, coordinator, response_generator, f"Person{i}", "")
    for i in range(5)
]
moderator = Moderator(None, None, coordinator, response_generator)


def test_unanimous_decision():
    mc = UnanimityConsensus(panelists[:3], use_moderator=False)
    agreements = [
        Agreement(
            agreement=False,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        )
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution="",
                agent_id="",
                persona="",
                response="",
                message_id=0,
            )
            for _ in range(2)
        ]
    )
    decision, is_consensus, agreements, voting_string, _ = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert is_consensus


def test_unanimous_decision_in_first_five_turns():
    mc = HybridMajorityConsensus(panelists[:3], use_moderator=False)
    agreements = [
        Agreement(
            agreement=False,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        )
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution="",
                agent_id="",
                persona="",
                response="",
                message_id=0,
            )
            for _ in range(2)
        ]
    )
    decision, is_consensus, agreements, voting_string, _ = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert is_consensus


def test_unanimous_decision_in_first_five_turns_with_moderator():
    mc = HybridMajorityConsensus(panelists[:3], use_moderator=True)
    agreements = [
        Agreement(
            agreement=False,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        ),
        Agreement(
            agreement=None,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        ),
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution="",
                agent_id="",
                persona="",
                response="",
                message_id=0,
            )
            for _ in range(3)
        ]
    )
    decision, is_consensus, agreements, voting_string, _ = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert is_consensus


def test_no_unanimous_decision_in_first_five_turns():
    mc = HybridMajorityConsensus(panelists[:3], use_moderator=False)
    agreements = [
        Agreement(
            agreement=False,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        ),
        Agreement(
            agreement=False,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        ),
        Agreement(
            agreement=True,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        ),
    ]
    decision, is_consensus, agreements, voting_string, _ = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert not is_consensus


def test_no_unanimous_decision_in_first_five_turns_with_moderator():
    mc = HybridMajorityConsensus(panelists[:2], use_moderator=True)
    agreements = [
        Agreement(
            agreement=None,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        ),
        Agreement(
            agreement=False,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        ),
        Agreement(
            agreement=True,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        ),
    ]
    decision, is_consensus, agreements, voting_string, _ = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert not is_consensus


def test_majority_decision_after_five_turns():
    mc = HybridMajorityConsensus(panelists, use_moderator=False)
    agreements = [
        Agreement(
            agreement=False,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        )
        for _ in range(2)
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution="",
                agent_id="",
                persona="",
                response="",
                message_id=0,
            )
            for _ in range(3)
        ]
    )
    decision, is_consensus, agreements, voting_string, _ = mc.make_decision(
        agreements, 6, 0, "", ""
    )
    assert is_consensus


def test_no_majority_decision_after_five_turns():
    mc = HybridMajorityConsensus(panelists, use_moderator=False)
    agreements = [
        Agreement(
            agreement=False,
            solution="",
            agent_id="",
            persona="",
            response="",
            message_id=0,
        )
        for _ in range(3)
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution="",
                agent_id="",
                persona="",
                response="",
                message_id=0,
            )
            for _ in range(2)
        ]
    )
    decision, is_consensus, agreements, voting_string, _ = mc.make_decision(
        agreements, 6, 0, "", ""
    )
    assert not is_consensus
