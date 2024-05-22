from typing import Any

import fire

from mallm.agents.agent import Agent
from mallm.utils.types import Agreement, TemplateFilling, Memory


class Panelist(Agent):
    def participate(
        self,
        use_moderator: bool,
        memories,
        unique_id,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        extract_all_drafts: bool,
        agents_to_update,
        agreements: list[Agreement],
    ) -> tuple[list[Memory], list[Agreement]]:
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
            )
        else:
            res, memory, agreements = self.improve(
                unique_id=unique_id,
                turn=turn,
                memory_ids=memory_ids,
                template_filling=template_filling,
                extract_all_drafts=extract_all_drafts,
                agreements=agreements,
            )

        memories.append(memory)
        memories = self.coordinator.update_memories(memories, agents_to_update)
        return memories, agreements


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
