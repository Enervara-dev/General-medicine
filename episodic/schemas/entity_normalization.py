"""
Entity normalisation — extractor output → the flat entity vocabulary the
episodic schema stores.

WHY
    The extraction LLM is free-form JSON. For a bare entity it emits a string
    ("weakness"); as soon as the entity carries a qualifier it upgrades the same
    slot to an object ({"name": "fever", "value": "101F"}). ``EpisodeEntities``
    declares ``list[str]``, so the object form failed validation and the WHOLE
    candidate was dropped — silently costing the turn its episode, and with it
    the downstream PMS memory event.

WHAT THIS DOES
    Coerces any reasonable entity representation into one display string, with
    the qualifier folded in rather than discarded:

        "weakness"                              -> "weakness"
        {"name": "fever", "value": "101F"}      -> "fever 101F"
        {"name": "cough", "severity": "mild"}   -> "cough mild"
        {"label": "BP", "reading": "140/90"}    -> "BP 140/90"

    It is deliberately generic: no condition-specific, symptom-specific or
    unit-specific handling. The same function serves every entity list
    (symptoms, conditions, medications, labs, body_parts) and any list added
    later, because it is applied to the model's fields as a group.

WHAT IS NOT LOST
    Folding into a string keeps the clinical detail on the wire to PMS, whose
    contract is ``tuple[str, ...]`` and is NOT changed here. The original
    structured objects are additionally preserved verbatim on
    ``EpisodeCandidate.metadata['entities_raw']`` — metadata is internal and
    never crosses the PMS boundary, so nothing is discarded and no contract
    moves.
"""

from __future__ import annotations

from typing import Any

# Keys that name the entity itself, in preference order.
_NAME_KEYS: tuple[str, ...] = (
    "name",
    "term",
    "label",
    "text",
    "title",
    "entity",
    "symptom",
    "condition",
    "medication",
    "lab",
    "body_part",
    "description",
)

# Keys that qualify the entity, in the order they should read. Anything not
# listed still survives (appended afterwards, key-sorted) — the list only fixes
# the ordering of the common cases so output is stable and readable.
_QUALIFIER_KEYS: tuple[str, ...] = (
    "value",
    "reading",
    "measurement",
    "amount",
    "dose",
    "dosage",
    "strength",
    "unit",
    "severity",
    "laterality",
    "site",
    "location",
    "status",
    "qualifier",
    "detail",
)

# Structural keys that carry no clinical meaning in the rendered string.
_IGNORED_KEYS: frozenset[str] = frozenset({"id", "code", "system", "type", "confidence"})

_MAX_ENTITY_CHARS = 200


def _scalar(value: Any) -> str:
    """Render a scalar as clean text; empty string for anything unusable."""
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    if isinstance(value, bool):  # a bare True/False qualifier says nothing useful
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def coerce_entity(item: Any) -> str:
    """
    Reduce one entity of any supported shape to a single display string.

    Returns "" when the item carries nothing usable, so callers can drop it.
    Never raises: this runs inside model validation on the chat path.
    """
    if isinstance(item, str):
        return item.strip()[:_MAX_ENTITY_CHARS]

    if not isinstance(item, dict):
        return _scalar(item)[:_MAX_ENTITY_CHARS]

    # 1) The name. Prefer a known naming key; otherwise fall back to the first
    #    usable scalar so an unexpected key shape still yields something.
    name = ""
    name_key = None
    for key in _NAME_KEYS:
        candidate = _scalar(item.get(key))
        if candidate:
            name, name_key = candidate, key
            break
    if not name:
        for key, value in item.items():
            if key in _IGNORED_KEYS:
                continue
            candidate = _scalar(value)
            if candidate:
                name, name_key = candidate, key
                break
    if not name:
        return ""

    # 2) Qualifiers: known ones first for stable reading order, then the rest by
    #    key so output is deterministic regardless of dict ordering.
    remaining = [
        k
        for k in item
        if k != name_key and k not in _IGNORED_KEYS and k not in _QUALIFIER_KEYS
    ]
    ordered_keys = [k for k in _QUALIFIER_KEYS if k in item and k != name_key]
    ordered_keys += sorted(remaining)

    seen_lower = {name.lower()}
    parts = [name]
    for key in ordered_keys:
        text = _scalar(item.get(key))
        if not text or text.lower() in seen_lower:
            continue  # empty, or merely restates the name
        seen_lower.add(text.lower())
        parts.append(text)

    return " ".join(parts)[:_MAX_ENTITY_CHARS]


def normalize_entity_list(value: Any) -> Any:
    """
    Normalise a whole entity list. Order-preserving and case-insensitively
    de-duplicated.

    Non-list input is returned untouched so pydantic can raise its own, clearer
    error rather than this function masking a genuinely malformed payload.
    """
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        value = [value]  # a lone entity where a list was expected
    if not isinstance(value, (list, tuple)):
        return value

    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        # The model occasionally nests a group; flatten one level.
        inner = item if isinstance(item, (list, tuple)) else [item]
        for element in inner:
            text = coerce_entity(element)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
    return out


def has_structured_entities(entities: Any) -> bool:
    """True when any entity list contains a non-string entry worth preserving."""
    if not isinstance(entities, dict):
        return False
    for value in entities.values():
        if isinstance(value, (list, tuple)):
            if any(not isinstance(v, str) for v in value):
                return True
        elif isinstance(value, dict):
            return True
    return False


__all__ = ["coerce_entity", "normalize_entity_list", "has_structured_entities"]
