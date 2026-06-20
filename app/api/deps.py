"""
FastAPI dependency-injection factories.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.container import AppContainer


def get_container(request: Request) -> AppContainer:
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="App is still starting; AppContainer not built yet.",
        )
    return container


ContainerDep = Annotated[AppContainer, Depends(get_container)]
