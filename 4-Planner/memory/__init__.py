from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


@dataclass
class MemoryTurn:
    user: str
    assistant: str
    final: Optional[Dict[str, object]] = None


class MemoryStore:
    def __init__(
        self,
        turns: Optional[List[MemoryTurn]] = None,
        *,
        max_turns: int = 6,
        max_chars: int = 1200,
        max_turn_chars: int = 360,
    ) -> None:
        self.turns: List[MemoryTurn] = list(turns or [])
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.max_turn_chars = max_turn_chars

    def add_turn(self, user: str, assistant: str, final: Optional[Dict[str, object]] = None) -> None:
        self.turns.append(MemoryTurn(user=user, assistant=assistant, final=final))

    def to_prompt(self) -> str:
        if not self.turns:
            return ""
        lines: List[str] = []
        for turn in self.turns[-self.max_turns :]:
            lines.append(f"User: {_truncate(turn.user, self.max_turn_chars)}")
            if turn.assistant:
                lines.append(f"Assistant: {_truncate(turn.assistant, self.max_turn_chars)}")
        prompt = "\n".join(lines)
        if len(prompt) > self.max_chars:
            prompt = prompt[-self.max_chars :]
        return prompt

    def to_list(self) -> List[Dict[str, object]]:
        return [
            {"user": turn.user, "assistant": turn.assistant, "final": turn.final or {}}
            for turn in self.turns
        ]

    @classmethod
    def from_list(cls, data: Optional[List[Dict[str, object]]]) -> "MemoryStore":
        turns: List[MemoryTurn] = []
        for item in data or []:
            if not isinstance(item, dict):
                continue
            user = str(item.get("user") or "")
            assistant = str(item.get("assistant") or "")
            final = item.get("final")
            final_dict = final if isinstance(final, dict) else None
            turns.append(MemoryTurn(user=user, assistant=assistant, final=final_dict))
        return cls(turns=turns)
