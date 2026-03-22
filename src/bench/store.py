"""Save/load benchmark results as JSON."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SystemInfo:
    chip: str = ""
    memory_gb: int = 0
    os_version: str = ""
    python_version: str = ""
    mlx_version: str = ""
    mlx_lm_version: str = ""

    @classmethod
    def detect(cls) -> SystemInfo:
        info = cls()
        info.os_version = platform.platform()
        info.python_version = platform.python_version()

        # Detect Apple Silicon chip
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            info.chip = result.stdout.strip()
        except Exception:
            info.chip = platform.processor()

        # Detect memory
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            info.memory_gb = int(result.stdout.strip()) // (1024 ** 3)
        except Exception:
            pass

        # Library versions
        try:
            import mlx

            info.mlx_version = mlx.__version__
        except Exception:
            pass
        try:
            import mlx_lm

            info.mlx_lm_version = mlx_lm.__version__
        except Exception:
            pass

        return info


@dataclass
class SessionResult:
    timestamp: str = ""
    duration_s: float = 0.0
    system_info: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    tool_calling: dict[str, Any] = field(default_factory=dict)
    aggregated: dict[str, Any] = field(default_factory=dict)
    power: dict[str, Any] = field(default_factory=dict)


def save_session(result: SessionResult, output_dir: str | Path) -> Path:
    """Save a session result to a JSON file. Returns the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = result.timestamp or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    filename = f"bench_{timestamp}.json"
    path = output_dir / filename

    with open(path, "w") as f:
        json.dump(asdict(result) if hasattr(result, "__dataclass_fields__") else _to_dict(result), f, indent=2, default=str)

    return path


def load_session(path: str | Path) -> SessionResult:
    """Load a session result from a JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    return SessionResult(
        timestamp=data.get("timestamp", ""),
        duration_s=data.get("duration_s", 0.0),
        system_info=data.get("system_info", {}),
        config_snapshot=data.get("config_snapshot", {}),
        runs=data.get("runs", []),
        quality=data.get("quality", {}),
        tool_calling=data.get("tool_calling", {}),
        aggregated=data.get("aggregated", {}),
        power=data.get("power", {}),
    )


def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
