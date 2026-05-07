"""AI SDR voice agent package."""

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from ai_sdr_agent.app import create_app as _create_app

        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
