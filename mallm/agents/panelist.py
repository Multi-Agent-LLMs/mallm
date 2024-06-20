from collections.abc import Sequence

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
    ) -> list[Agreement]:
        """
        Either calls feedback() or improve() depending on whether a moderator is present
        """
        if use_moderator:
            _res, memory, agreements = self.feedback(
                unique_id=unique_id,
                turn=turn,
                memory_ids=memory_ids,
                template_filling=template_filling,
                agreements=agreements,
            )
        else:
            _res, memory, agreements = self.improve(
                unique_id=unique_id,
                turn=turn,
                memory_ids=memory_ids,
                template_filling=template_filling,
                agreements=agreements,
            )

        memories.append(memory)
        self.coordinator.update_memories(memories, agents_to_update)
        return agreements
