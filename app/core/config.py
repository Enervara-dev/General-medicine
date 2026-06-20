"""
Thin facade over graphrag.config.settings — re-exports the project Settings
singleton so the app/ package never imports outside its own boundary except
through this module.
"""

from graphrag.config.settings import ConfigError, Settings, settings


def get_settings() -> Settings:
    """Dependency-injection accessor for the singleton Settings."""
    return settings


__all__ = ["ConfigError", "Settings", "get_settings", "settings"]
