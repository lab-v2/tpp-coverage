#!/usr/bin/env python3
"""
Environment file loading utilities.
"""

from pathlib import Path


def load_env_file(env_path: str = ".env") -> dict[str, str]:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file

    Returns:
        Dictionary of environment variables
    """
    env_vars: dict[str, str] = {}
    env_file = Path(env_path)

    if not env_file.exists():
        print(f"Warning: Environment file {env_path} not found")
        return env_vars

    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse KEY=VALUE format
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

        print(f"Info: Loaded {len(env_vars)} environment variables from {env_path}")
        return env_vars

    except (OSError, PermissionError) as e:
        print(f"Warning: Error reading environment file {env_path}: {e}")
        return env_vars
