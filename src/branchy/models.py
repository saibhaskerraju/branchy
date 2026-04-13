from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MessageNode(BaseModel):
    message_id: str
    user: str
    assistant: str
    parent_message_id: Optional[str]
    child_message_ids: list[str] = Field(default_factory=list)
