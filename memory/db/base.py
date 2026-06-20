"""
SQLAlchemy declarative base with explicit constraint naming.

Predictable constraint names matter for Alembic autogenerate diffs and for
operability (you can reference a constraint by name without inspecting it).
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Project-wide SQLAlchemy declarative base."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
