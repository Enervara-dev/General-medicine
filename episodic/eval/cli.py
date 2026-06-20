"""
Eval CLI: label retrieval events and compute aggregate ranking metrics.

Usage:
    python -m episodic.eval.cli list                 # show all logged events
    python -m episodic.eval.cli label                # interactive labeling loop
    python -m episodic.eval.cli label <event_id>     # label one event
    python -m episodic.eval.cli metrics              # aggregate precision@k / MRR / nDCG
    python -m episodic.eval.cli metrics --k 3        # use k=3 instead of default 5
    python -m episodic.eval.cli show <event_id>      # show one event in detail

Labels are stored in EPISODIC_EVAL_LABELS_PATH (data/eval/labels.jsonl by
default), one record per event_id. Re-labeling an event appends a new
record; the latest record wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from episodic.config import EpisodicConfig
from episodic.eval.metrics import aggregate


# Windows console fix — emojis + box-drawing chars otherwise crash cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_latest_labels(path: Path) -> dict[str, dict]:
    """Collapse an append-only labels JSONL into {event_id: latest_label}."""
    latest: dict[str, dict] = {}
    for row in _load_jsonl(path):
        eid = row.get("event_id")
        if eid:
            latest[eid] = row
    return latest


def _append_label(path: Path, event_id: str, relevant_ids: list[str], notes: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "event_id": event_id,
        "labeled_at": datetime.now(tz=timezone.utc).isoformat(),
        "relevant_episode_ids": relevant_ids,
        "notes": notes,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_event(event: dict) -> None:
    print(f"\nevent_id : {event['event_id']}")
    print(f"timestamp: {event['timestamp']}")
    print(f"user_id  : {event['user_id']}")
    print(f"query    : {event['query_text']}")
    print(f"results  : ({len(event.get('results', []))} ranked)")
    for r in event.get("results", []):
        print(
            f"  [{r['rank']:2d}] {r['episode_id']}  score={r['score']:.3f}  "
            f"[{r.get('category', '-')}]"
        )
        print(f"        {r.get('summary', '')[:120]}")


def cmd_list(args: argparse.Namespace) -> None:
    events = _load_jsonl(Path(EpisodicConfig.EVAL_LOG_PATH))
    labels = _load_latest_labels(Path(EpisodicConfig.EVAL_LABELS_PATH))
    if not events:
        print(f"No events logged. Set EPISODIC_EVAL_LOGGING_ENABLED=true and run queries first.")
        print(f"Expected log file: {EpisodicConfig.EVAL_LOG_PATH}")
        return
    print(f"{'event_id':36s}  {'labeled':7s}  {'user':16s}  query")
    for ev in events:
        eid = ev["event_id"]
        marker = "yes" if eid in labels else "—"
        user = ev.get("user_id", "")[:16]
        query = (ev.get("query_text") or "")[:60]
        print(f"{eid:36s}  {marker:7s}  {user:16s}  {query}")


def cmd_show(args: argparse.Namespace) -> None:
    events = _load_jsonl(Path(EpisodicConfig.EVAL_LOG_PATH))
    target = next((e for e in events if e["event_id"] == args.event_id), None)
    if target is None:
        print(f"No event with id {args.event_id}")
        sys.exit(1)
    _print_event(target)
    labels = _load_latest_labels(Path(EpisodicConfig.EVAL_LABELS_PATH))
    lbl = labels.get(args.event_id)
    if lbl:
        print(f"\nlabel    : relevant_ids = {lbl.get('relevant_episode_ids')}")
        if lbl.get("notes"):
            print(f"           notes        = {lbl['notes']}")


def cmd_label(args: argparse.Namespace) -> None:
    events = _load_jsonl(Path(EpisodicConfig.EVAL_LOG_PATH))
    labels = _load_latest_labels(Path(EpisodicConfig.EVAL_LABELS_PATH))
    if not events:
        print("No events logged. Nothing to label.")
        return

    if args.event_id:
        events = [e for e in events if e["event_id"] == args.event_id]
        if not events:
            print(f"No event with id {args.event_id}")
            sys.exit(1)
    else:
        # By default, only show unlabeled events to skip what's already done.
        events = [e for e in events if e["event_id"] not in labels]
        if not events:
            print("All events are already labeled. Use `metrics` to see scores.")
            return

    for ev in events:
        _print_event(ev)
        print(
            "\nEnter relevant episode_id(s), comma-separated. "
            "Tip: use the rank numbers (1,2,3) and we'll map them. "
            "Press Enter to skip, Ctrl-C to quit."
        )
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLabeling stopped.")
            return

        if not raw:
            print("(skipped)")
            continue

        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        ranked = ev.get("results", [])
        relevant: list[str] = []
        for tok in tokens:
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(ranked):
                    relevant.append(ranked[idx]["episode_id"])
            else:
                relevant.append(tok)

        notes = ""
        try:
            notes = input("notes (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            pass

        _append_label(
            Path(EpisodicConfig.EVAL_LABELS_PATH),
            event_id=ev["event_id"],
            relevant_ids=relevant,
            notes=notes,
        )
        print(f"saved label for {ev['event_id']}: {relevant}")


def cmd_metrics(args: argparse.Namespace) -> None:
    events = _load_jsonl(Path(EpisodicConfig.EVAL_LOG_PATH))
    labels = _load_latest_labels(Path(EpisodicConfig.EVAL_LABELS_PATH))
    if not events:
        print("No events logged.")
        return
    pairs: list[dict] = []
    for ev in events:
        lbl = labels.get(ev["event_id"])
        if not lbl:
            continue
        pairs.append(
            {
                "retrieved": [r["episode_id"] for r in ev.get("results", [])],
                "relevant": lbl.get("relevant_episode_ids") or [],
            }
        )

    if not pairs:
        print("No labeled events yet. Run `python -m episodic.eval.cli label` first.")
        return

    summary = aggregate(pairs, k=args.k)
    print("\nEpisodic Memory — Ranking Eval")
    print("-" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.3f}")
        else:
            print(f"  {k:20s} = {v}")
    print("-" * 60)
    print(f"  log file   : {EpisodicConfig.EVAL_LOG_PATH}")
    print(f"  labels file: {EpisodicConfig.EVAL_LABELS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Episodic memory eval CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Show all logged events + whether they're labeled.")

    p_show = sub.add_parser("show", help="Show one event in detail.")
    p_show.add_argument("event_id")

    p_label = sub.add_parser("label", help="Interactively label one or all events.")
    p_label.add_argument("event_id", nargs="?", default=None)

    p_metrics = sub.add_parser("metrics", help="Aggregate precision@k / MRR / nDCG.")
    p_metrics.add_argument("--k", type=int, default=5)

    args = parser.parse_args()
    {"list": cmd_list, "show": cmd_show, "label": cmd_label, "metrics": cmd_metrics}[args.cmd](args)


if __name__ == "__main__":
    main()
