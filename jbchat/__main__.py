"""Run the JB-Chat web interface:  python -m jbchat"""

from __future__ import annotations

import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run("jbchat.web:create_app", factory=True,
                 host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()