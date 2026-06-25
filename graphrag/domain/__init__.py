"""Domain helpers shared across stacks (canned answer blocks, terminal rule, …)."""

from graphrag.domain.messages import (
    emergency_blocks,
    is_terminal_turn,
    out_of_scope_blocks,
    refusal_blocks,
)

__all__ = [
    "emergency_blocks",
    "is_terminal_turn",
    "out_of_scope_blocks",
    "refusal_blocks",
]
