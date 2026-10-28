"""
eval.py — a lightweight QA harness for the triage classifier.

This is the "measure quality, don't hope for it" piece: it runs every ticket
in data/tickets.json (which carries a human-assigned ground-truth label)
through the classifier, then reports:

  - overall category accuracy, plus per-category precision/recall
  - a confusion matrix, so you can see exactly what gets confused with what
  - urgency accuracy
  - most important: the "silent failure" count — tickets the model got
    WRONG that also weren't caught by the human-review safety net. That
    number is the one a support ops lead actually cares about, because it's
    the volume that would have gone out the door unsupervised and wrong.

Usage:
  python eval.py                # uses data/eval_cache.json if present
  python eval.py --refresh      # re-runs every ticket through the classifier
  python eval.py --refresh --model claude-opus-4-1-20250805
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from triage import triage, CATEGORIES

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
TICKETS_PATH = DATA_DIR / "tickets.json"
CACHE_PATH = DATA_DIR / "eval_cache.json"


def run_classifications(force_refresh: bool, model: str | None):
    tickets = json.loads(TICKETS_PATH.read_text())

    if CACHE_PATH.exists() and not force_refresh:
        cached = json.loads(CACHE_PATH.read_text())
        if len(cached) == len(tickets):
            return tickets, cached

    predictions = []
    for t in tickets:
        r = triage(t, model=model)
        predictions.append({
            "ticket_id": r.ticket_id,
            "predicted_category": r.classification.category,
            "predicted_urgency": r.classification.urgency,
            "confidence": r.classification.confidence,
            "route": r.route,
            "needs_human_review": r.needs_human_review,
            "source": r.classification.source,
        })

    CACHE_PATH.write_text(json.dumps(predictions, indent=2))
    return tickets, predictions


def build_report(tickets, predictions):
    by_id = {t["id"]: t for t in tickets}

    total = len(tickets)
    category_correct = 0
    urgency_correct = 0

    confusion = defaultdict(lambda: defaultdict(int))  # true -> predicted -> count
    per_category_true = defaultdict(int)
    per_category_predicted_correct = defaultdict(int)
    per_category_predicted_total = defaultdict(int)

    silent_failures = []   # wrong category AND not flagged for human review
    caught_failures = []   # wrong category BUT flagged for human review

    for p in predictions:
        t = by_id[p["ticket_id"]]
        true_cat = t["true_category"]
        true_urg = t["true_urgency"]
        pred_cat = p["predicted_category"]
        pred_urg = p["predicted_urgency"]

        confusion[true_cat][pred_cat] += 1
        per_category_true[true_cat] += 1
        per_category_predicted_total[pred_cat] += 1

        is_correct = pred_cat == true_cat
        if is_correct:
            category_correct += 1
            per_category_predicted_correct[pred_cat] += 1
        else:
            record = {
                "ticket_id": p["ticket_id"],
                "subject": t["subject"],
                "true_category": true_cat,
                "predicted_category": pred_cat,
                "confidence": p["confidence"],
                "route": p["route"],
            }
            if p["needs_human_review"]:
                caught_failures.append(record)
            else:
                silent_failures.append(record)

        if pred_urg == true_urg:
            urgency_correct += 1

    per_category_stats = {}
    for cat in CATEGORIES:
        true_n = per_category_true.get(cat, 0)
        pred_n = per_category_predicted_total.get(cat, 0)
        correct_n = per_category_predicted_correct.get(cat, 0)
        recall = correct_n / true_n if true_n else None
        precision = correct_n / pred_n if pred_n else None
        per_category_stats[cat] = {
            "support": true_n,
            "predicted_count": pred_n,
            "precision": precision,
            "recall": recall,
        }

    return {
        "total_tickets": total,
        "category_accuracy": category_correct / total if total else 0,
        "urgency_exact_accuracy": urgency_correct / total if total else 0,
        "per_category": per_category_stats,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "silent_failures": silent_failures,
        "caught_failures": caught_failures,
        "safety_net_catch_rate": (
            len(caught_failures) / (len(caught_failures) + len(silent_failures))
            if (caught_failures or silent_failures) else None
        ),
    }


def print_report(report, source_note: str):
    print("=" * 64)
    print("TRIAGE CLASSIFIER — QA REPORT")
    print(f"({source_note})")
    print("=" * 64)
    print(f"Tickets evaluated:        {report['total_tickets']}")
    print(f"Category accuracy:        {report['category_accuracy']*100:.1f}%")
    print(f"Urgency exact-match acc:  {report['urgency_exact_accuracy']*100:.1f}%")
    print()

    print("Per-category (support = # true examples in the eval set):")
    print(f"  {'Category':<20}{'Support':>8}{'Precision':>12}{'Recall':>10}")
    for cat, stats in report["per_category"].items():
        prec = f"{stats['precision']*100:.0f}%" if stats["precision"] is not None else "—"
        rec = f"{stats['recall']*100:.0f}%" if stats["recall"] is not None else "—"
        print(f"  {cat:<20}{stats['support']:>8}{prec:>12}{rec:>10}")
    print()

    n_wrong = len(report["silent_failures"]) + len(report["caught_failures"])
    print(f"Misclassified tickets:    {n_wrong}")
    if report["safety_net_catch_rate"] is not None:
        print(f"  caught by human-review safety net: {len(report['caught_failures'])} "
              f"({report['safety_net_catch_rate']*100:.0f}%)")
        print(f"  SILENT failures (wrong + auto-routed): {len(report['silent_failures'])}")
    if report["silent_failures"]:
        print()
        print("  ⚠ These would have gone out the door unsupervised and wrong:")
        for f in report["silent_failures"]:
            print(f"    [{f['ticket_id']}] \"{f['subject']}\" — true={f['true_category']}, "
                  f"predicted={f['predicted_category']}, route={f['route']}")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="QA eval for the ticket triage classifier.")
    parser.add_argument("--refresh", action="store_true",
                         help="Re-run classification instead of using the cached predictions.")
    parser.add_argument("--model", default=None,
                         help="Override the model (defaults to TRIAGE_MODEL env var / triage.py default).")
    args = parser.parse_args()

    tickets, predictions = run_classifications(force_refresh=args.refresh, model=args.model)
    report = build_report(tickets, predictions)

    used_cache = CACHE_PATH.exists() and not args.refresh
    source_note = f"from cache: {CACHE_PATH.name}" if used_cache else "freshly classified"
    print_report(report, source_note)

    report_path = DATA_DIR / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {report_path}")


if __name__ == "__main__":
    main()
