"""Domain helpers shared across stacks (canned answer blocks, terminal rule, …)."""

from graphrag.domain.messages import (
    canned_blocks_for,
    emergency_blocks,
    is_terminal_turn,
    mental_health_crisis_blocks,
    out_of_scope_blocks,
    parse_diagnostic_confidence,
    refusal_blocks,
)

__all__ = [
    "canned_blocks_for",
    "emergency_blocks",
    "is_terminal_turn",
    "mental_health_crisis_blocks",
    "out_of_scope_blocks",
    "parse_diagnostic_confidence",
    "refusal_blocks",
]
