from collections.abc import Sequence

from mallm.agents.agent import Agent
from mallm.utils.types import Agreement, Memory, TemplateFilling


class Panelist(Agent):
    """
    Represents a Panelist agent, a type of Agent that specializes in participating in discussions.
    It is prompted to either improve the current solution or provide feedback.
    """
    def participate(
        self,
        memories: list[Memory],
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        agents_to_update: Sequence[Agent],
        agreements: list[Agreement],
    ) -> list[Agreement]:
        """
        Either calls feedback() or improve(), depending on whether this panelist is allowed to draft solutions or not.
        """
        if self.drafting_agent:
            _res, memory, agreements = self.improve(
                unique_id=unique_id,
                turn=turn,
                memory_ids=memory_ids,
                template_filling=template_filling,
                agreements=agreements,
            )
        else:
            _res, memory, agreements = self.feedback(
                unique_id=unique_id,
                turn=turn,
                memory_ids=memory_ids,
                template_filling=template_filling,
                agreements=agreements,
            )

        memories.append(memory)
        self.coordinator.update_memories(memories, agents_to_update)
        return agreements
