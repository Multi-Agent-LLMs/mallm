from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.coordinator import Coordinator
from mallm.decision_protocol.majority import HybridMajorityConsensus
from mallm.utils.types import Agreement

coordinator = Coordinator(None, None)
panelists = [Panelist(None, None, coordinator, f"Person{i}", "") for i in range(5)]
moderator = Moderator(None, None, coordinator)


def test_unanimous_decision_in_first_five_turns():
    mc = HybridMajorityConsensus(panelists[:3], use_moderator=False)
    agreements = [Agreement(agreement=False, agent_id="", persona="", response="")]
    agreements.extend(
        [
            Agreement(agreement=True, agent_id="", persona="", response="")
            for _ in range(2)
        ]
    )
    decision, is_consensus = mc.make_decision(agreements, 4, 0, "", "")
    assert is_consensus


def test_unanimous_decision_in_first_five_turns_with_moderator():
    mc = HybridMajorityConsensus(panelists[:3], use_moderator=True)
    agreements = [Agreement(agreement=False, agent_id="", persona="", response="")]
    agreements.append(Agreement(agreement=None, agent_id="", persona="", response=""))
    agreements.extend(
        [
            Agreement(agreement=True, agent_id="", persona="", response="")
            for _ in range(3)
        ]
    )
    decision, is_consensus = mc.make_decision(agreements, 4, 0, "", "")
    assert is_consensus


def test_no_unanimous_decision_in_first_five_turns():
    mc = HybridMajorityConsensus(panelists[:3], use_moderator=False)
    agreements = [
        Agreement(agreement=False, agent_id="", persona="", response=""),
        Agreement(agreement=False, agent_id="", persona="", response=""),
        Agreement(agreement=True, agent_id="", persona="", response=""),
    ]
    decision, is_consensus = mc.make_decision(agreements, 4, 0, "", "")
    assert not is_consensus


def test_no_unanimous_decision_in_first_five_turns_with_moderator():
    mc = HybridMajorityConsensus(panelists[:2], use_moderator=True)
    agreements = [
        Agreement(agreement=None, agent_id="", persona="", response=""),
        Agreement(agreement=False, agent_id="", persona="", response=""),
        Agreement(agreement=True, agent_id="", persona="", response=""),
    ]
    decision, is_consensus = mc.make_decision(agreements, 4, 0, "", "")
    assert not is_consensus


def test_majority_decision_after_five_turns():
    mc = HybridMajorityConsensus(panelists, use_moderator=False)
    agreements = [
        Agreement(agreement=False, agent_id="", persona="", response="")
        for _ in range(2)
    ]
    agreements.extend(
        [
            Agreement(agreement=True, agent_id="", persona="", response="")
            for _ in range(3)
        ]
    )
    decision, is_consensus = mc.make_decision(agreements, 6, 0, "", "")
    assert is_consensus


def test_no_majority_decision_after_five_turns():
    mc = HybridMajorityConsensus(panelists, use_moderator=False)
    agreements = [
        Agreement(agreement=False, agent_id="", persona="", response="")
        for _ in range(3)
    ]
    agreements.extend(
        [
            Agreement(agreement=True, agent_id="", persona="", response="")
            for _ in range(2)
        ]
    )
    decision, is_consensus = mc.make_decision(agreements, 6, 0, "", "")
    assert not is_consensus