from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TicketStatus = Literal["open", "pending", "resolved"]
TicketSentiment = Literal["positive", "neutral", "negative"]


class TicketCreate(BaseModel):
    subject: str = Field(min_length=3, max_length=200)
    message: str = Field(min_length=10, max_length=10000)


class TicketStatusUpdate(BaseModel):
    status: TicketStatus


class AnalysisResult(BaseModel):
    category: str
    sentiment: TicketSentiment
    urgency: int = Field(ge=1, le=5)
    summary: str
    suggested_response: str
    tags: list[str]


class TicketRead(BaseModel):
    id: int
    subject: str
    message: str
    status: TicketStatus
    category: str | None = None
    sentiment: TicketSentiment | None = None
    urgency: int | None = Field(default=None, ge=1, le=5)
    summary: str | None = None
    suggested_response: str | None = None
    tags: list[str] | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)