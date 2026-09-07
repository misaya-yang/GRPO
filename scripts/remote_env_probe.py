"""Read-only runtime inventory; use the selected research Python."""

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

packages = {}
for name in ("torch", "transformers", "numpy", "accelerate", "safetensors", "peft", "datasets"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
print(
    json.dumps(
        {
            "observed_utc": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "executable": sys.executable,
            "architecture": platform.machine(),
            "packages": packages,
            "gpu": gpu.stdout.strip(),
            "cgroup_memory": {
                str(path): path.read_text().strip()
                for path in (
                    Path("/sys/fs/cgroup/memory.max"),
                    Path("/sys/fs/cgroup/memory.high"),
                    Path("/sys/fs/cgroup/memory.current"),
                )
                if path.exists()
            },
        },
        indent=2,
    )
)
