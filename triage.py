"""
triage.py — core AI triage engine for support tickets.

This module is deliberately split into two layers, which is the main design
idea of this project:

1. classify_ticket()   -> asks an LLM ONLY to understand the ticket
                           (category, urgency, confidence, a suggested reply).
2. apply_routing_policy() -> a plain-Python rules layer that decides what
                           happens next (self-serve, AI auto-reply, human
                           agent, escalate to engineering/finance).

Keeping these separate means the *judgment call* of "where does automation
end and a human need to step in" lives in code a support lead can read and
tune directly, instead of being buried inside a prompt. That separation is
the thing I'd actually want to defend in an interview.

If no ANTHROPIC_API_KEY is set, classify_ticket() falls back to a small
keyword-based heuristic classifier (see _mock_classify) so the rest of the
app — routing, the eval harness, the UI — can be exercised without an API
key. The heuristic is intentionally simple; it exists for demo/offline use,
not as a real classifier.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

CATEGORIES = [
    "Billing",
    "Bug Report",
    "Product Question",
    "Account/Onboarding",
    "Feature Request",
    "Other",
]

URGENCY_LEVELS = ["Low", "Medium", "High", "Urgent"]

CONFIDENCE_THRESHOLD = 0.70

SYSTEM_PROMPT = """You are a triage assistant for the customer support team at \
Instrumentl, a SaaS platform that helps nonprofits discover, track, and win \
grant funding. You read one incoming support ticket and classify it. You do \
NOT decide how it gets routed — a separate policy layer handles that. Your \
only job is to understand the ticket accurately.

Respond with ONLY a JSON object (no markdown fences, no commentary) with \
exactly these fields:
{
  "category": one of ["Billing", "Bug Report", "Product Question", "Account/Onboarding", "Feature Request", "Other"],
  "urgency": one of ["Low", "Medium", "High", "Urgent"],
  "confidence": a number from 0.0 to 1.0 representing how confident you are in the category classification,
  "reasoning": one short sentence explaining the classification,
  "suggested_reply": a short, warm, professional draft reply (2-4 sentences) a support agent could send as a starting point, written for a non-technical nonprofit user. If the ticket involves money, an outage, or account access, the draft should acknowledge the issue and set an expectation rather than promise a specific fix.
}

Judge urgency by real impact: a blocked deadline, an outage, a duplicate \
charge, or lost data is High or Urgent. A general question or a nice-to-have \
feature request is Low or Medium."""


@dataclass
class Classification:
    category: str
    urgency: str
    confidence: float
    reasoning: str
    suggested_reply: str
    source: str = "llm"  # "llm" or "mock"


@dataclass
class TriageResult:
    ticket_id: str
    subject: str
    classification: Classification
    route: str
    needs_human_review: bool
    routing_reason: str


def _get_client():
    """Lazily import/construct the Anthropic client so this module can be
    imported (e.g. by eval.py in mock mode) without the package installed."""
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def _extract_json(text: str) -> dict:
    """Model responses are asked to be raw JSON, but strip code fences
    defensively in case the model wraps it anyway."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    return json.loads(text)


def classify_ticket(subject: str, body: str, model: Optional[str] = None) -> Classification:
    """Classify a single ticket. Uses the Claude API if ANTHROPIC_API_KEY is
    set, otherwise falls back to a heuristic mock classifier."""
    client = _get_client()
    if client is None:
        return _mock_classify(subject, body)

    model = model or os.environ.get("TRIAGE_MODEL", "claude-haiku-4-5")
    ticket_text = f"Subject: {subject}\n\nBody: {body}"

    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": ticket_text}],
    )
    raw_text = response.content[0].text
    try:
        data = _extract_json(raw_text)
        return Classification(
            category=data["category"],
            urgency=data["urgency"],
            confidence=float(data["confidence"]),
            reasoning=data.get("reasoning", ""),
            suggested_reply=data.get("suggested_reply", ""),
            source="llm",
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        # A real support-ops system should never silently drop a ticket it
        # couldn't parse — force it to a human instead of guessing.
        return Classification(
            category="Other",
            urgency="High",
            confidence=0.0,
            reasoning=f"Could not parse model output ({exc}); routed to a human as a safety net.",
            suggested_reply="",
            source="llm-parse-error",
        )


def _mock_classify(subject: str, body: str) -> Classification:
    """Offline fallback: cheap keyword heuristics. Good enough to exercise
    routing/UI logic without an API key; not a substitute for the real thing."""
    text = f"{subject} {body}".lower()

    # Strip common negations before keyword-matching so "not urgent" / "no rush"
    # don't trip the same trigger as an actual urgent request.
    negated_text = re.sub(r"\b(not|no|isn't|without any)\s+(urgent|rush|hurry)\b", "", text)

    urgent_words = ["urgent", "asap", "immediately", "right away", "emergency", "today", "in an hour"]
    high_words = ["deadline", "duplicate", "charged twice", "down", "500 error", "can't log in",
                  "cannot log in", "outage", "disappeared", "vanished", "lost", "blocked"]

    if any(w in text for w in ["refund", "charge", "invoice", "billed", "billing", "subscription",
                                "downgrade", "upgrade", "cancel", "plan"]):
        category, base_conf = "Billing", 0.6
    elif any(w in text for w in ["error", "bug", "broke", "broken", "fail", "crash", "500",
                                  "disappeared", "vanished", "slow", "down", "wrong date"]):
        category, base_conf = "Bug Report", 0.6
    elif any(w in text for w in ["invite", "login", "log in", "password", "add a team member",
                                  "onboarding", "onboard", "seat"]):
        category, base_conf = "Account/Onboarding", 0.55
    elif any(w in text for w in ["would love", "feature request", "suggestion", "nice to have",
                                  "it would help", "request:"]):
        category, base_conf = "Feature Request", 0.55
    elif any(w in text for w in ["how do", "how does", "question", "difference between",
                                  "integrate", "does instrumentl"]):
        category, base_conf = "Product Question", 0.6
    else:
        category, base_conf = "Other", 0.4

    if any(w in negated_text for w in urgent_words):
        urgency = "Urgent"
    elif any(w in negated_text for w in high_words):
        urgency = "High"
    elif category in ("Feature Request",):
        urgency = "Low"
    else:
        urgency = "Medium"

    return Classification(
        category=category,
        urgency=urgency,
        confidence=base_conf,
        reasoning="Heuristic keyword match (mock mode — no ANTHROPIC_API_KEY set).",
        suggested_reply=(
            "Thanks for reaching out — we've received your message and a member of our "
            "team will follow up shortly. (This is a placeholder reply generated in mock mode.)"
        ),
        source="mock",
    )


def apply_routing_policy(classification: Classification, subject: str, body: str) -> tuple[str, bool, str]:
    """Decide where a classified ticket goes. This is plain, readable code on
    purpose — it's the layer a support ops lead should be able to change
    without touching a prompt. Returns (route, needs_human_review, reason).

    Routes:
      - "self_serve"          -> point to a help center article, no reply needed
      - "ai_auto_reply"       -> the AI's suggested_reply can go out as-is
      - "human_agent"         -> needs a person to read and respond
      - "escalate_engineering"-> looks like a real product bug
      - "escalate_finance"    -> money is involved and needs a finance-aware human
    """
    text = f"{subject} {body}".lower()
    category = classification.category
    urgency = classification.urgency
    confidence = classification.confidence

    # Safety net: low-confidence classifications always go to a human,
    # regardless of what category/urgency the model guessed.
    if confidence < CONFIDENCE_THRESHOLD:
        return ("human_agent", True,
                f"Confidence {confidence:.2f} is below the {CONFIDENCE_THRESHOLD} threshold.")

    # Safety net: anything the model itself flagged Urgent gets human eyes,
    # even if the category-specific rule below would have auto-replied.
    if urgency == "Urgent":
        return ("human_agent", True, "Urgency is Urgent — always routed to a human.")

    if category == "Billing":
        money_words = ["refund", "duplicate", "charged twice", "cancel", "chargeback"]
        if urgency in ("High", "Urgent") or any(w in text for w in money_words):
            return ("escalate_finance", True,
                    "Billing ticket involving refunds/cancellations or elevated urgency — needs a finance-aware human.")
        return ("human_agent", True, "Billing tickets always get a human reply, even routine ones.")

    if category == "Bug Report":
        if urgency in ("High", "Urgent"):
            return ("escalate_engineering", True,
                    "High-impact bug report — needs reproduction and an engineering look.")
        return ("human_agent", False,
                "Lower-impact bug report — a support agent can gather repro steps first.")

    if category == "Account/Onboarding":
        return ("human_agent", False, "Account changes touch access/security — kept human-in-the-loop.")

    if category == "Feature Request":
        return ("ai_auto_reply", False,
                "Low-stakes — AI can thank the customer and log it to the product backlog.")

    if category == "Product Question":
        if confidence >= 0.85:
            return ("self_serve", False, "High-confidence product question — likely answerable from the Help Center.")
        return ("ai_auto_reply", False, "Product question — AI draft is a good starting point for a quick human check.")

    return ("human_agent", True, "Unrecognized category — defaulting to a human.")


def triage(ticket: dict, model: Optional[str] = None) -> TriageResult:
    """Full pipeline for one ticket dict with 'id'/'subject'/'body' keys."""
    classification = classify_ticket(ticket["subject"], ticket["body"], model=model)
    route, needs_review, reason = apply_routing_policy(classification, ticket["subject"], ticket["body"])
    return TriageResult(
        ticket_id=ticket.get("id", ""),
        subject=ticket["subject"],
        classification=classification,
        route=route,
        needs_human_review=needs_review,
        routing_reason=reason,
    )
