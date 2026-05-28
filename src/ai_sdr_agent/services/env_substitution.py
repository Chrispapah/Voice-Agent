from __future__ import annotations

import re
from typing import Mapping

_ENV_PATTERN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


def substitute_env_vars(text: str, env: Mapping[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            raise KeyError(f"Missing workspace environment variable: {name}")
        return env[name]

    return _ENV_PATTERN.sub(repl, text)


def substitute_env_vars_optional(text: str, env: Mapping[str, str]) -> str:
    try:
        return substitute_env_vars(text, env)
    except KeyError:
        return text
