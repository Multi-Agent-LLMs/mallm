from mallm.agents.draftProposer import DraftProposer
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
draft_proposer = DraftProposer(None, None, coordinator, response_generator)


def test_unanimous_decision():
    mc = UnanimityConsensus(panelists[:3], num_neutral_agents=0)
    agreements = [
        Agreement(
            agreement=False,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        )
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution=None,
                agent_id="",
                persona="",
                response="",
                message_id="",
            )
            for _ in range(2)
        ]
    )
    decision, is_consensus, agreements, voting_string = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert is_consensus


def test_unanimous_decision_in_first_five_turns():
    mc = HybridMajorityConsensus(panelists[:3], num_neutral_agents=0)
    agreements = [
        Agreement(
            agreement=False,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        )
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution=None,
                agent_id="",
                persona="",
                response="",
                message_id="",
            )
            for _ in range(2)
        ]
    )
    decision, is_consensus, agreements, voting_string = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert is_consensus


def test_unanimous_decision_in_first_five_turns_with_draft_proposer():
    mc = HybridMajorityConsensus(panelists[:3], num_neutral_agents=0)
    agreements = [
        Agreement(
            agreement=False,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        ),
        Agreement(
            agreement=None,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        ),
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution=None,
                agent_id="",
                persona="",
                response="",
                message_id="",
            )
            for _ in range(2)
        ]
    )
    decision, is_consensus, agreements, voting_string = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert is_consensus


def test_no_unanimous_decision_in_first_five_turns():
    mc = HybridMajorityConsensus(panelists[:3], num_neutral_agents=0)
    agreements = [
        Agreement(
            agreement=False,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        ),
        Agreement(
            agreement=False,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        ),
        Agreement(
            agreement=True,
            solution=None,
            agent_id="",
            persona="",
            response="",
            message_id="",
        ),
    ]
    decision, is_consensus, agreements, voting_string = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert not is_consensus


def test_no_unanimous_decision_in_first_five_turns_with_draft_proposer():
    mc = HybridMajorityConsensus(panelists[:2], num_neutral_agents=1)
    agreements = [
        Agreement(
            agreement=None,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        ),
        Agreement(
            agreement=False,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        ),
        Agreement(
            agreement=True,
            solution=None,
            agent_id="",
            persona="",
            response="",
            message_id="",
        ),
    ]
    decision, is_consensus, agreements, voting_string = mc.make_decision(
        agreements, 4, 0, "", ""
    )
    assert not is_consensus

def test_majority_decision_after_five_turns():
    mc = HybridMajorityConsensus(panelists, num_neutral_agents=0)
    agreements = [
        Agreement(
            agreement=False,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        )
        for _ in range(2)
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution=None,
                agent_id="",
                persona="",
                response="",
                message_id="",
            )
            for _ in range(3)
        ]
    )
    decision, is_consensus, agreements, voting_string = mc.make_decision(
        agreements, 6, 0, "", ""
    )
    assert is_consensus


def test_no_majority_decision_after_five_turns():
    mc = HybridMajorityConsensus(panelists, num_neutral_agents=0)
    agreements = [
        Agreement(
            agreement=False,
            solution="Test",
            agent_id="",
            persona="",
            response="",
            message_id="",
        )
        for _ in range(3)
    ]
    agreements.extend(
        [
            Agreement(
                agreement=True,
                solution=None,
                agent_id="",
                persona="",
                response="",
                message_id="",
            )
            for _ in range(2)
        ]
    )
    decision, is_consensus, agreements, voting_string = mc.make_decision(
        agreements, 6, 0, "", ""
    )
    assert not is_consensus
