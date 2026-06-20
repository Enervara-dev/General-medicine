"""
Seed CLI for the episodic memory layer.

Usage:
    python -m episodic.scripts.seed --persona cardio_test
    python -m episodic.scripts.seed --all
    python -m episodic.scripts.seed --all --no-wipe   # add to existing without clearing

Reads fixture JSON files from episodic/fixtures/ and upserts EpisodeCandidate
entries directly into Pinecone via PineconeEpisodicRepository. Skips the
extraction LLM so seeding is deterministic + cheap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from pinecone.errors.exceptions import NotFoundError

from episodic.api.dependencies import build_container
from episodic.schemas.episode import Episode, EpisodeCandidate
from episodic.services.storage import PineconeEpisodicRepository

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
ALL_PERSONAS = ("cardio_test", "migraine_test", "geriatric_test")


def _load_fixture(persona: str) -> tuple[str, list[Episode]]:
    """Read a persona fixture JSON and convert each entry into a timestamped Episode."""
    path = FIXTURES_DIR / f"{persona}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    user_id = data["user_id"]
    now = datetime.now(tz=timezone.utc)

    episodes: list[Episode] = []
    for entry in data["episodes"]:
        days_ago = int(entry.pop("_days_ago", 0))
        # Inject user_id from the file header so the JSON entries don't repeat it.
        entry.setdefault("user_id", user_id)
        candidate = EpisodeCandidate.model_validate(entry)
        ep = Episode.from_candidate(candidate).model_copy(
            update={
                "episode_id": uuid.uuid4(),
                "timestamp": now - timedelta(days=days_ago),
            }
        )
        episodes.append(ep)
    return user_id, episodes


async def _wipe_namespace(
    repository: PineconeEpisodicRepository,
    user_id: str,
    existing_episodes: Iterable[Episode],  # kept for signature symmetry; unused
) -> int:
    """
    Delete every vector currently in the user's namespace.

    Returns the count wiped (-1 if Pinecone doesn't report it). A 404 means
    the namespace doesn't exist yet — first-run case — so treat it as 0.
    """
    index = repository._get_index()
    try:
        await asyncio.to_thread(index.delete, namespace=user_id, delete_all=True)
        return -1  # unknown count, but the namespace was reachable and emptied
    except NotFoundError:
        # Namespace doesn't exist yet — nothing to wipe, this is benign.
        return 0


async def _seed_persona(
    repository: PineconeEpisodicRepository,
    persona: str,
    *,
    wipe: bool,
) -> None:
    user_id, episodes = _load_fixture(persona)
    print(f"\n=== {persona} ({len(episodes)} episodes) ===")

    if wipe:
        wiped = await _wipe_namespace(repository, user_id, episodes)
        if wiped == -1:
            wiped_str = "all prior vectors"
        elif wiped == 0:
            wiped_str = "namespace empty / first run"
        else:
            wiped_str = f"{wiped} prior vectors"
        print(f"  - wiped namespace ({wiped_str})")

    await repository.upsert_batch(episodes)
    print(f"  - upserted {len(episodes)} episodes into namespace '{user_id}'")


async def _main_async(personas: list[str], *, wipe: bool) -> None:
    container = build_container()
    await container.repository.ensure_index()
    for persona in personas:
        await _seed_persona(container.repository, persona, wipe=wipe)
    print("\nSeed complete.\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Seed the episodic memory layer with persona fixtures.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--persona", choices=ALL_PERSONAS, help="Seed a single persona.")
    group.add_argument("--all", action="store_true", help="Seed all three personas.")
    parser.add_argument(
        "--no-wipe",
        action="store_true",
        help="Do NOT clear the namespace before upserting. Default: wipe first.",
    )

    args = parser.parse_args()
    personas = list(ALL_PERSONAS) if args.all else [args.persona]
    wipe = not args.no_wipe

    asyncio.run(_main_async(personas, wipe=wipe))


if __name__ == "__main__":
    main()
