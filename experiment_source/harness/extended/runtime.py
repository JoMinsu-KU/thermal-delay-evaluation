"""Original H200/jev checks and shared lock; no model training."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from validation_io import require
REQUIRED_PREFIX=Path("/home/jovyan/.conda/envs/jev")

def utc():
    return datetime.now(timezone.utc).isoformat()

def environment_check(root):
    require(sys.platform.startswith("linux"), "Experiments must run on the original Linux H200 instance")
    require(Path(sys.prefix).resolve() == REQUIRED_PREFIX.resolve(), f"Activate Conda {REQUIRED_PREFIX}")
    for key in ("JEV_LOAD_8BIT", "JEV_LOAD_4BIT", "JEV_DEVICE_MAP", "JEV_MAX_MEMORY"):
        require(not os.environ.get(key), f"Unset {key}; this protocol preserves original BF16/CUDA placement")
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Use the supplied launcher for offline model loading")
    expected_cache = (root / "cache/huggingface/hub").resolve()
    require(Path(os.environ.get("HF_HUB_CACHE", "")).resolve() == expected_cache, "HF_HUB_CACHE differs from the archived project cache")
    for key in ("TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        require(not os.environ.get(key) or Path(os.environ[key]).resolve() == expected_cache, f"{key} overrides the verified model cache")
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one visible CUDA GPU is required")
    gpu = torch.cuda.get_device_properties(0)
    require("H200" in gpu.name, f"Original protocol requires H200; found {gpu.name}")
    packages = {}
    for package in ("torch", "numpy", "scipy", "transformers", "peft", "accelerate", "safetensors", "PyYAML", "huggingface-hub", "tokenizers"):
        packages[package] = importlib.metadata.version(package)
    result = {
        "checked_at_utc": utc(), "python": sys.version, "prefix": sys.prefix,
        "platform": platform.platform(), "packages": packages, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": gpu.name,
        "gpu_total_memory_gib": gpu.total_memory / 2**30,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PYTHONHASHSEED",
            "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE")},
    }
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True)
    result["nvidia_smi"] = query.stdout.strip() or query.stderr.strip()
    result["driver_version"] = query.stdout.split(",")[1].strip() if query.returncode == 0 and "," in query.stdout else "unknown"
    return result

def environment_identity(env):
    return {key: value for key, value in env.items() if key not in ("checked_at_utc", "nvidia_smi")}

@contextmanager
def experiment_lock(root):
    import fcntl
    path = root / "locks/l0.lock"
    require(path.parent.is_dir(), f"Missing original lock directory: {path.parent}")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another L0/L1/L2 process holds locks/l0.lock; stop this run and retry later") from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "started_utc": utc(), "purpose": "extended_numerical_validation"}) + "\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
