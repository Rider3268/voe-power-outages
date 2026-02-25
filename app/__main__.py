from __future__ import annotations

import uvicorn

from app.config import load_settings


if __name__ == "__main__":
    settings = load_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
