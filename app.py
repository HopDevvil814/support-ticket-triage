"""
app.py — small Flask demo UI for the triage engine in triage.py.

Two things to try:
  /            paste a ticket, see how it gets classified and routed
  /queue       runs triage over the sample ticket set (data/tickets.json)
               and shows it as a support queue would see it, with a
               summary of how much volume would be handled without a human.

Results for the sample queue are cached to data/triage_cache.json so you
aren't re-billing API calls on every page refresh. Delete that file (or hit
/queue?refresh=1) to force a re-run.
"""

import json
import os
from pathlib import Path

from flask import Flask, render_template, request
from dotenv import load_dotenv

from triage import triage, CATEGORIES, URGENCY_LEVELS

load_dotenv()

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data"
TICKETS_PATH = DATA_DIR / "tickets.json"
CACHE_PATH = DATA_DIR / "triage_cache.json"

ROUTE_LABELS = {
    "self_serve": ("Self-serve", "good"),
    "ai_auto_reply": ("AI auto-reply", "good"),
    "human_agent": ("Human agent", "warning"),
    "escalate_engineering": ("Escalate: Engineering", "serious"),
    "escalate_finance": ("Escalate: Finance", "critical"),
}


def _load_tickets():
    with open(TICKETS_PATH) as f:
        return json.load(f)


def _result_to_dict(result):
    return {
        "ticket_id": result.ticket_id,
        "subject": result.subject,
        "category": result.classification.category,
        "urgency": result.classification.urgency,
        "confidence": round(result.classification.confidence, 2),
        "reasoning": result.classification.reasoning,
        "suggested_reply": result.classification.suggested_reply,
        "source": result.classification.source,
        "route": result.route,
        "route_label": ROUTE_LABELS.get(result.route, (result.route, "warning"))[0],
        "route_status": ROUTE_LABELS.get(result.route, (result.route, "warning"))[1],
        "needs_human_review": result.needs_human_review,
        "routing_reason": result.routing_reason,
    }


def _run_queue(force_refresh: bool = False):
    if CACHE_PATH.exists() and not force_refresh:
        with open(CACHE_PATH) as f:
            return json.load(f)

    tickets = _load_tickets()
    results = [_result_to_dict(triage(t)) for t in tickets]

    with open(CACHE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    subject = ""
    body = ""

    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        body = request.form.get("body", "").strip()
        if not subject or not body:
            error = "Please fill in both a subject and a body."
        else:
            r = triage({"id": "ADHOC", "subject": subject, "body": body})
            result = _result_to_dict(r)

    mock_mode = not bool(os.environ.get("ANTHROPIC_API_KEY"))
    return render_template(
        "index.html",
        result=result,
        error=error,
        subject=subject,
        body=body,
        mock_mode=mock_mode,
    )


@app.route("/queue")
def queue():
    force_refresh = request.args.get("refresh") == "1"
    results = _run_queue(force_refresh=force_refresh)

    total = len(results)
    auto_handled = sum(1 for r in results if r["route"] in ("self_serve", "ai_auto_reply"))
    needs_review = sum(1 for r in results if r["needs_human_review"])
    by_category = {}
    for r in results:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1

    mock_mode = not bool(os.environ.get("ANTHROPIC_API_KEY"))
    return render_template(
        "queue.html",
        results=results,
        total=total,
        auto_handled=auto_handled,
        needs_review=needs_review,
        auto_pct=round(100 * auto_handled / total) if total else 0,
        by_category=by_category,
        mock_mode=mock_mode,
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=int(os.environ.get("PORT", 5000)))
