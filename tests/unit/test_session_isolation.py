"""Cross-user session isolation — user-scoped Redis keys.

Forces the SessionManager into its in-memory fallback (no Redis needed) so the
key-scoping logic is exercised deterministically.
"""

from __future__ import annotations

import pytest

from Memory_Layer.session_memory.session_manager import (
    SessionManager,
    _IN_MEMORY_STORE,
    _redis_key,
)
from Memory_Layer.session_memory.models import Message, Role, SessionMemory


@pytest.fixture
def mgr():
    m = SessionManager()
    m._use_fallback = True  # deterministic: skip Redis, use the RAM store
    _IN_MEMORY_STORE.clear()
    yield m
    _IN_MEMORY_STORE.clear()


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def test_key_is_user_scoped_when_user_id_present():
    assert _redis_key("S1", "userA") == "session:userA:S1"


def test_key_is_flat_when_anonymous():
    assert _redis_key("S1") == "session:S1"
    assert _redis_key("S1", None) == "session:S1"
    assert _redis_key("S1", "") == "session:S1"


# ---------------------------------------------------------------------------
# Cross-user isolation
# ---------------------------------------------------------------------------

async def _save(mgr, session_id, user_id, marker):
    s = SessionMemory(session_id=session_id)
    s.add_turn(Message(role=Role.USER, content=marker))
    await mgr.save_session(s, user_id=user_id)


async def test_two_users_same_session_id_are_isolated(mgr):
    # Both users use the SAME session_id but different user_ids.
    await _save(mgr, "SHARED", "userA", "A-secret")
    await _save(mgr, "SHARED", "userB", "B-secret")

    a = await mgr.load_session("SHARED", user_id="userA")
    b = await mgr.load_session("SHARED", user_id="userB")

    assert a is not None and b is not None
    assert a.recent_turns[0].content == "A-secret"
    assert b.recent_turns[0].content == "B-secret"  # NOT A's data


async def test_user_cannot_read_another_users_session(mgr):
    await _save(mgr, "S1", "userA", "A-only")
    # userB presenting userA's session_id gets a MISS, not A's memory.
    assert await mgr.load_session("S1", user_id="userB") is None
    # userA still reads their own.
    assert (await mgr.load_session("S1", user_id="userA")).recent_turns[0].content == "A-only"


async def test_authenticated_and_anonymous_are_separate(mgr):
    # Same session_id, one authenticated, one anonymous → different keys.
    await _save(mgr, "S1", "userA", "auth")
    await _save(mgr, "S1", None, "anon")
    assert (await mgr.load_session("S1", user_id="userA")).recent_turns[0].content == "auth"
    assert (await mgr.load_session("S1")).recent_turns[0].content == "anon"


async def test_anonymous_session_roundtrip_unchanged(mgr):
    # Backward-compat: no user_id behaves exactly as before.
    await _save(mgr, "S1", None, "hello")
    loaded = await mgr.load_session("S1")
    assert loaded is not None and loaded.recent_turns[0].content == "hello"
    assert "session:S1" in _IN_MEMORY_STORE  # flat key


async def test_delete_is_user_scoped(mgr):
    await _save(mgr, "S1", "userA", "A")
    await _save(mgr, "S1", "userB", "B")
    # Deleting A's does not touch B's.
    assert await mgr.delete_session("S1", user_id="userA") is True
    assert await mgr.load_session("S1", user_id="userA") is None
    assert await mgr.load_session("S1", user_id="userB") is not None


async def test_stored_session_id_is_unchanged_client_facing(mgr):
    # The scoping affects only the storage KEY, never the object's session_id
    # that the client persists and re-sends.
    await _save(mgr, "S1", "userA", "x")
    loaded = await mgr.load_session("S1", user_id="userA")
    assert loaded.session_id == "S1"


# ---------------------------------------------------------------------------
# App-layer wrapper threads user_id
# ---------------------------------------------------------------------------

async def test_app_layer_load_and_save_scope_by_user(mgr):
    from app.services.memory.session import load_session, save_after_turn

    # Save a turn for userA via the app-layer helper.
    bundle = await load_session(mgr, "S1", user_id="userA")
    await save_after_turn(
        mgr,
        session=bundle.session,
        user_query="I have a fever",
        assistant_answer="ok",
        analysis={"intent": "symptom_query"},
        query_type="symptom_query",
        user_id="userA",
    )
    # userB with the same session_id starts fresh (no turns).
    other = await load_session(mgr, "S1", user_id="userB")
    assert other.session.total_messages == 0
    # userA sees their saved turns.
    mine = await load_session(mgr, "S1", user_id="userA")
    assert mine.session.total_messages > 0
