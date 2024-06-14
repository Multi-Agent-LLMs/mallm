from typing import Sequence

from mallm.agents.agent import Agent
from mallm.utils.types import Agreement, Memory, TemplateFilling


class Panelist(Agent):
    def participate(
        self,
        use_moderator: bool,
        memories: list[Memory],
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        agents_to_update: Sequence[Agent],
        agreements: list[Agreement],
        chain_of_thought: bool,
    ) -> list[Agreement]:
        """
        Either calls feedback() or improve() depending on whether a moderator is present
        """
        if use_moderator:
            res, memory, agreements = self.feedback(
                unique_id=unique_id,
                turn=turn,
                memory_ids=memory_ids,
                template_filling=template_filling,
                agreements=agreements,
                chain_of_thought=chain_of_thought,
            )
        else:
            res, memory, agreements = self.improve(
                unique_id=unique_id,
                turn=turn,
                memory_ids=memory_ids,
                template_filling=template_filling,
                agreements=agreements,
                chain_of_thought=chain_of_thought,
            )

        memories.append(memory)
        self.coordinator.update_memories(memories, agents_to_update)
        return agreements
