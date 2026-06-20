"""
Scenario runner CLI for the episodic memory demo harness.

Usage:
    python -m episodic.scripts.scenarios --persona cardio_test
    python -m episodic.scripts.scenarios --all

For each scenario, prints:
  - the label and the feature it exercises
  - the expected_behavior line (what a reviewer should look for)
  - a compact view of the actual service response
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

# Windows console default codepage (cp1252) can't render the box-drawing
# characters this script uses. Force stdout/stderr to UTF-8 so the report
# prints cleanly on PowerShell.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from episodic.api.dependencies import EpisodicContainer, build_container
from episodic.schemas.clarification import ClarificationRequest
from episodic.schemas.retrieval import RetrievalRequest
from episodic.scripts._scenarios_data import (
    SCENARIOS_BY_PERSONA,
    Scenario,
)

logger = logging.getLogger(__name__)


HR = "─" * 88
DHR = "═" * 88


async def _run_one(
    container: EpisodicContainer,
    persona: str,
    scenario: Scenario,
) -> str:
    """Dispatch one scenario to the matching service and format the response."""
    payload = {**scenario.payload, "user_id": persona}
    kind = scenario.kind

    try:
        if kind == "retrieve":
            req = RetrievalRequest.model_validate(payload)
            ranked = await container.retriever.retrieve(req)
            return _format_ranked(ranked)

        if kind == "context":
            req = RetrievalRequest.model_validate(payload)
            block = await container.context_pipeline.build(req)
            return _format_context(block)

        if kind == "clarify":
            req = ClarificationRequest.model_validate(payload)
            resp = await container.clarifier.evaluate(
                user_id=req.user_id,
                utterance=req.utterance,
            )
            return _format_clarification(resp)

        if kind == "contradictions":
            new_claim = payload["new_claim"]
            top_k = payload.get("top_k", 10)
            ranked = await container.retriever.retrieve(
                RetrievalRequest(
                    user_id=persona, query_text=new_claim, top_k=top_k, return_k=top_k
                )
            )
            report = await container.contradiction.detect(
                user_id=persona,
                new_claim=new_claim,
                prior_episodes=[r.episode for r in ranked],
            )
            return _format_contradictions(report)

        if kind == "store":
            result = await container.ingest_pipeline.run(
                user_id=persona,
                utterance=payload["utterance"],
            )
            return _format_ingest(result)

        return f"  ! unknown scenario kind: {kind}"
    except Exception as exc:
        logger.exception("Scenario '%s' failed: %s", scenario.label, exc)
        return f"  ! ERROR: {exc.__class__.__name__}: {exc}"


def _format_ranked(ranked) -> str:
    if not ranked:
        return "  (no episodes returned)"
    lines = []
    for i, r in enumerate(ranked, start=1):
        ts = r.episode.timestamp.date()
        cat = r.episode.category.value
        prio = r.episode.clinical_priority.value
        sev = r.episode.severity.value
        f = r.factors
        lines.append(
            f"  [{i}] score={r.score:.3f}  sim={f.get('similarity', 0):.2f}  "
            f"rec={f.get('recency', 0):.2f}  prio={f.get('priority', 0):.2f}  "
            f"conf={f.get('confidence', 0):.2f}  rcr={f.get('recurrence', 0):.2f}"
        )
        lines.append(f"        {ts} [{cat}/{prio}/{sev}] {r.episode.summary}")
    return "\n".join(lines)


def _format_context(block) -> str:
    lines = [
        f"  strategy: {block.metadata.get('strategy', '-')}",
        f"  raw_count={block.metadata.get('raw_count', 0)}  "
        f"kept_count={block.metadata.get('kept_count', 0)}  "
        f"compressed_count={block.metadata.get('compressed_count', 0)}",
    ]
    if block.compressed:
        lines.append("  --- COMPRESSED CLUSTERS ---")
        for c in block.compressed:
            lines.append(
                f"  • [{c.category}] members={len(c.member_ids)} "
                f"window={c.first_seen.date()}→{c.last_seen.date()} "
                f"peak={c.peak_severity}"
            )
            lines.append(f"    {c.summary}")
    if block.episodes:
        lines.append("  --- INDIVIDUAL EPISODES (top 5) ---")
        for r in block.episodes[:5]:
            ts = r.episode.timestamp.date()
            lines.append(
                f"  • score={r.score:.3f}  {ts}  [{r.episode.category.value}]  "
                f"{r.episode.summary}"
            )
    if not block.episodes and not block.compressed:
        lines.append("  (no episodes returned)")
    if block.rendered_prompt:
        lines.append("  --- RENDERED PROMPT BLOCK ---")
        for ln in block.rendered_prompt.splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines)


def _format_clarification(resp) -> str:
    lines = [f"  needs_clarification: {resp.needs_clarification}"]
    for q in resp.questions:
        lines.append(
            f"  • [{q.reason.value}] safety_critical={q.safety_critical}"
        )
        lines.append(f"    Q: {q.question}")
    return "\n".join(lines)


def _format_contradictions(report) -> str:
    lines = [
        f"  has_contradictions: {report.has_contradictions}",
        f"  confidence_penalty: {report.confidence_penalty:.2f}",
        f"  triggers_clarification: {report.triggers_clarification}",
    ]
    for c in report.contradictions:
        lines.append(f"  • severity={c.severity.value}")
        lines.append(f"    prior   : {c.prior_summary}")
        lines.append(f"    current : {c.current_claim}")
        lines.append(f"    reason  : {c.reason}")
    return "\n".join(lines)


def _format_ingest(result) -> str:
    lines = []
    if result.stored is not None:
        ep = result.stored
        lines.append(f"  stored episode_id={ep.episode_id}")
        lines.append(f"    category={ep.category.value}  priority={ep.clinical_priority.value}  "
                     f"severity={ep.severity.value}  confidence={ep.confidence:.2f}")
        lines.append(f"    summary: {ep.summary}")
        lines.append(f"    embedding_text: {ep.embedding_text}")
    else:
        lines.append("  stored: (none — extraction returned empty OR clarification required)")
        if result.candidate is not None:
            lines.append(f"    candidate summary: {result.candidate.summary}")
    lines.append(
        f"  clarification: needs={result.clarification.needs_clarification} "
        f"questions={len(result.clarification.questions)}"
    )
    for q in result.clarification.questions:
        lines.append(f"    Q: {q.question}")
    lines.append(
        f"  contradictions: has={result.contradictions.has_contradictions} "
        f"penalty={result.contradictions.confidence_penalty:.2f}"
    )
    return "\n".join(lines)


async def _run_persona(container: EpisodicContainer, persona: str) -> None:
    scenarios = SCENARIOS_BY_PERSONA.get(persona)
    if not scenarios:
        print(f"  ! no scenarios defined for {persona}")
        return

    print(f"\n{DHR}\n  PERSONA: {persona}  ({len(scenarios)} scenarios)\n{DHR}")
    for idx, sc in enumerate(scenarios, start=1):
        print(f"\n{HR}")
        print(f"  [{idx}/{len(scenarios)}] {sc.label}  ({sc.feature})")
        print(f"  expected: {sc.expected_behavior}")
        if sc.payload:
            compact = {k: v for k, v in sc.payload.items() if k != "user_id"}
            print(f"  payload : {json.dumps(compact, default=str)}")
        print(HR)
        out = await _run_one(container, persona, sc)
        print(out)


async def _main_async(personas: list[str]) -> None:
    container = build_container()
    await container.repository.ensure_index()
    for persona in personas:
        await _run_persona(container, persona)
    print()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)  # quiet by default; scenarios print their own output

    parser = argparse.ArgumentParser(description="Run the episodic memory scenario suite.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--persona",
        choices=list(SCENARIOS_BY_PERSONA.keys()),
        help="Run scenarios for a single persona.",
    )
    group.add_argument("--all", action="store_true", help="Run scenarios for all personas.")

    args = parser.parse_args()
    personas = list(SCENARIOS_BY_PERSONA.keys()) if args.all else [args.persona]

    asyncio.run(_main_async(personas))


if __name__ == "__main__":
    main()
