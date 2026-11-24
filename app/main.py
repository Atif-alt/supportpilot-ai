from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from openai import OpenAIError
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai_service import analyze_ticket
from app.database import Base, engine, get_db
from app.models import Ticket
from app.schemas import TicketCreate, TicketRead, TicketStatusUpdate


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="SupportPilot AI",
    description="Support-ticket APIs with AI-assisted analysis and reply drafts.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    index_file = Path(__file__).parent / "static" / "index.html"

    if not index_file.is_file():
        return HTMLResponse(
            content=(
                "<h1>SupportPilot AI</h1>"
                "<p>Frontend file is missing: app/static/index.html</p>"
                "<p><a href='/docs'>Open the API documentation</a></p>"
            ),
            status_code=404,
        )

    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


def get_ticket_or_404(ticket_id: int, db: Session) -> Ticket:
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return ticket


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/tickets", response_model=TicketRead, status_code=201)
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    ticket = Ticket(subject=payload.subject, message=payload.message)
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


@app.get("/api/v1/tickets", response_model=list[TicketRead])
def list_tickets(
    status: Literal["open", "pending", "resolved"] | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    statement = select(Ticket)

    if status:
        statement = statement.where(Ticket.status == status)

    statement = (
        statement
        .order_by(Ticket.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    return list(db.scalars(statement).all())


@app.get("/api/v1/tickets/{ticket_id}", response_model=TicketRead)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    return get_ticket_or_404(ticket_id, db)


@app.post("/api/v1/tickets/{ticket_id}/analyze", response_model=TicketRead)
def analyze_support_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = get_ticket_or_404(ticket_id, db)

    try:
        result = analyze_ticket(ticket.subject, ticket.message)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="Configure Azure OpenAI before requesting ticket analysis.",
        ) from exc
    except (OpenAIError, ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Azure OpenAI analysis failed. Check the deployment settings and retry.",
        ) from exc

    ticket.category = result.category
    ticket.sentiment = result.sentiment
    ticket.urgency = result.urgency
    ticket.summary = result.summary
    ticket.suggested_response = result.suggested_response
    ticket.tags = result.tags

    db.commit()
    db.refresh(ticket)
    return ticket


@app.patch("/api/v1/tickets/{ticket_id}/status", response_model=TicketRead)
def update_ticket_status(
    ticket_id: int,
    payload: TicketStatusUpdate,
    db: Session = Depends(get_db),
):
    ticket = get_ticket_or_404(ticket_id, db)
    ticket.status = payload.status
    db.commit()
    db.refresh(ticket)
    return ticket