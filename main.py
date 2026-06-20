import argparse
import sys
from uuid import uuid4


def _print_banner(
    session_id: str,
    redis_url: str | None,
    show_memory: bool,
    user_id: str | None,
) -> None:
    print("\nEnervera Medical GraphRAG")
    print("=" * 72)
    print(f"Session memory : Redis-backed")
    print(f"Session id     : {session_id}")
    print(f"Redis URL      : {redis_url or 'REDIS_URL env / redis://localhost:6379/0'}")
    print(f"Episodic user  : {user_id or '(disabled — pass --user-id to enable)'}")
    print(f"Memory display : {'on' if show_memory else 'off'}")
    print("-" * 72)
    print("Commands: :memory  :session  :help  quit")
    print("=" * 72 + "\n")


def _compact_list(values: list[str], limit: int = 8) -> str:
    clean = [str(v).strip() for v in values if str(v).strip()]
    if not clean:
        return "-"
    shown = clean[:limit]
    suffix = f" (+{len(clean) - limit} more)" if len(clean) > limit else ""
    return ", ".join(shown) + suffix


def _print_memory_snapshot(pipeline, session_id: str) -> None:
    try:
        bundle = pipeline.memory_adapter.load(session_id)
    except Exception as exc:
        print("\n[Memory snapshot unavailable]")
        print(f"Reason: {exc}\n")
        return

    session = bundle.session
    state = bundle.working_memory.state

    print("\nSession Memory Snapshot")
    print("-" * 72)
    print(f"Session id          : {session.session_id}")
    print(f"Recent turns        : {len(session.recent_turns)}")
    print(f"Rolling summary     : {'yes' if bool(session.summary.strip()) else 'no'}")
    print(f"Symptoms            : {_compact_list(state.symptoms)}")
    print(f"Medications         : {_compact_list(state.drugs)}")
    print(f"Allergies           : {_compact_list(state.allergies)}")
    print(f"Conditions          : {_compact_list(state.conditions)}")
    print(f"Chronic conditions  : {_compact_list(state.chronic_conditions)}")
    print(f"Durations           : {_compact_list(state.duration)}")
    print(f"Previous concerns   : {_compact_list(state.previous_concerns)}")
    print(f"Follow-up refs      : {_compact_list(state.follow_up_references, limit=3)}")
    print(f"Discussed entities  : {_compact_list(state.discussed_entities)}")
    print(f"Active task         : {state.active_task or '-'}")
    print(f"Risk level          : {state.risk_level or '-'}")
    print("-" * 72 + "\n")


def _handle_command(command: str, pipeline, session_id: str) -> bool:
    cmd = command.strip().lower()
    if cmd in {":help", "help"}:
        print("\nAvailable commands")
        print("- :memory   Show the Redis-backed clinical memory snapshot")
        print("- :session  Show the current session id")
        print("- quit      Exit the assistant\n")
        return True
    if cmd == ":memory":
        _print_memory_snapshot(pipeline, session_id)
        return True
    if cmd == ":session":
        print(f"\nCurrent session id: {session_id}\n")
        return True
    return False


def run_query(
    query_text: str | None = None,
    session_id: str = "default",
    redis_url: str | None = None,
    show_memory: bool = True,
    user_id: str | None = None,
) -> None:
    from graphrag.config.settings import ConfigError, settings

    try:
        settings.validate_required("cli")
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        sys.exit(2)

    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline

    try:
        pipeline = GraphRAGPipeline(redis_url=redis_url)
    except Exception as e:
        print(f"Failed to initialize GraphRAG Engine: {e}")
        sys.exit(1)

    _print_banner(session_id, redis_url, show_memory, user_id)

    try:
        if query_text:
            print("Patient question")
            print("-" * 72)
            print(query_text)
            print("-" * 72 + "\n")
            pipeline.run(query_text, session_id=session_id, user_id=user_id)
            if show_memory:
                _print_memory_snapshot(pipeline, session_id)
            return

        while True:
            try:
                user_input = input("Ask a medical question: ").strip()
                if user_input.lower() in {"quit", "exit", "q", ""}:
                    break
                if _handle_command(user_input, pipeline, session_id):
                    continue

                print("\nRunning memory-aware GraphRAG...\n")
                pipeline.run(user_input, session_id=session_id, user_id=user_id)
                if show_memory:
                    _print_memory_snapshot(pipeline, session_id)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}\n")
    finally:
        print("\nShutting down gracefully...")
        pipeline.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enervera Medical GraphRAG CLI")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Optional: run one medical question and exit.",
    )
    parser.add_argument(
        "--session-id",
        default="default",
        help="Patient/session identifier used for Redis-backed conversation memory.",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Generate a fresh session id for this run.",
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis connection URL. Defaults to REDIS_URL or redis://localhost:6379/0.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help=(
            "Pinecone-episodic user_id (namespace). When supplied, the pipeline "
            "loads that user's episodic memory before the LLM call and ingests "
            "each user turn after the answer. Examples: cardio_test, "
            "migraine_test, geriatric_test."
        ),
    )
    parser.add_argument(
        "--hide-memory",
        action="store_true",
        help="Do not print the clinical memory snapshot after each answer.",
    )

    args = parser.parse_args()
    active_session_id = uuid4().hex if args.new_session else args.session_id

    run_query(
        args.query,
        session_id=active_session_id,
        redis_url=args.redis_url,
        show_memory=not args.hide_memory,
        user_id=args.user_id,
    )
