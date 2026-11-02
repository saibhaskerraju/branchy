from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MessageNode(BaseModel):
    message_id: str
    content: str
    node_type: Literal["user", "assistant"]
    parent_message_id: Optional[str]
    child_message_ids: list[str] = Field(default_factory=list)
    created_seq: int = 0
    created_at: str = ""
    tentative: bool = False
    routing: Optional[dict] = None


class Sticker(BaseModel):
    root_id: str
    text: str
    last_used_seq: int = 0
