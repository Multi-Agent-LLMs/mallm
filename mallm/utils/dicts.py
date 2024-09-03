from mallm.decision_protocol.approval import ApprovalVoting
from mallm.decision_protocol.cumulative import CumulativeVoting
from mallm.decision_protocol.majority import (
    HybridMajorityConsensus,
    MajorityConsensus,
    SupermajorityConsensus,
    UnanimityConsensus,
)
from mallm.decision_protocol.protocol import DecisionProtocol
from mallm.decision_protocol.ranked import RankedVoting
from mallm.decision_protocol.voting import Voting
from mallm.discourse_policy.debate import DiscourseDebate
from mallm.discourse_policy.memory import DiscourseMemory
from mallm.discourse_policy.policy import DiscoursePolicy
from mallm.discourse_policy.relay import DiscourseRelay
from mallm.discourse_policy.report import DiscourseReport
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.models.discussion.JSONResponseGenerator import JSONResponseGenerator
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.models.discussion.SimpleResponseGenerator import SimpleResponseGenerator
from mallm.models.discussion.SplitFreeTextResponseGenerator import (
    SplitFreeTextResponseGenerator,
)
from mallm.models.personas.ExpertGenerator import ExpertGenerator
from mallm.models.personas.IPIPPersonaGenerator import IPIPPersonaGenerator
from mallm.models.personas.MockGenerator import MockGenerator
from mallm.models.personas.NoPersonaGenerator import NoPersonaGenerator
from mallm.models.personas.PersonaGenerator import PersonaGenerator

DECISION_PROTOCOLS: dict[str, type[DecisionProtocol]] = {
    "majority_consensus": MajorityConsensus,
    "supermajority_consensus": SupermajorityConsensus,
    "hybrid_consensus": HybridMajorityConsensus,
    "unanimity_consensus": UnanimityConsensus,
    "voting": Voting,
    "approval": ApprovalVoting,
    "cumulative": CumulativeVoting,
    "ranked": RankedVoting,
}

DISCUSSION_PARADIGMS: dict[str, type[DiscoursePolicy]] = {
    "memory": DiscourseMemory,
    "report": DiscourseReport,
    "relay": DiscourseRelay,
    "debate": DiscourseDebate,
}

PERSONA_GENERATORS: dict[str, type[PersonaGenerator]] = {
    "expert": ExpertGenerator,
    "ipip": IPIPPersonaGenerator,
    "nopersona": NoPersonaGenerator,
    "mock": MockGenerator,
}

RESPONSE_GENERATORS: dict[str, type[ResponseGenerator]] = {
    "json": JSONResponseGenerator,
    "freetext": FreeTextResponseGenerator,
    "splitfreetext": SplitFreeTextResponseGenerator,
    "simple": SimpleResponseGenerator,
}
