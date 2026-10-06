from __future__ import annotations

"""
ATHAR OS — RunPod Serverless SAFE ENTRYPOINT

Purpose
-------
Keep the RunPod worker process alive and healthy BEFORE any heavy model,
PEFT adapter, Git-LFS, or advisory module work starts.

Heavy imports/model loading are intentionally lazy and happen only after an
actual advisory_consultation job is assigned to the worker.

This file is intentionally focused on the current production Screen-3 route:
    type == "advisory_consultation"

It also exposes a zero-model preflight:
    {"input": {"type": "advisory_consultation", "preflight": true}}
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import runpod


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent

# Production repo path used by the existing Athar image.  If that path is not
# present (for example in a different build layout), fall back to the directory
# containing this handler.py.
_DEFAULT_ROOT = Path("/workspace/data/athar")
ROOT = Path(
    os.getenv(
        "ATHAR_ROOT",
        str(_DEFAULT_ROOT if _DEFAULT_ROOT.exists() else _THIS_DIR),
    )
).resolve()

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-14B").strip()

GITHUB_REPO = os.getenv(
    "GITHUB_REPO",
    "da02bod-art/athar-qwen-training",
).strip()
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main").strip()

SPECIALIST_REL = os.getenv(
    "COUNCIL_SPECIALIST_REL",
    "checkpoints/specialist",
).strip("/")
META_REL = os.getenv(
    "COUNCIL_META_REL",
    "checkpoints/meta",
).strip("/")

_ENGINE = None
_ENGINE_LOCK = threading.RLock()

# Keep references to the shared model/tokenizer so they are not garbage
# collected and so every subsequent request reuses the warm worker.
_BASE_MODEL = None
_TOKENIZER = None

BOOT_VERSION = "athar-runpod-safe-entry-v2-lfs-recovery"


# ---------------------------------------------------------------------------
# Small startup helpers — stdlib only
# ---------------------------------------------------------------------------

def _log(message: str) -> None:
    print(
        f"[ATHAR_BOOT] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}",
        flush=True,
    )


def _run(
    cmd,
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            + proc.stdout[-8000:]
        )
    return proc.stdout


def _real_adapter_checkpoint(path: Path) -> bool:
    """
    Git-LFS pointer files are tiny. A real adapter_model.safetensors in this
    project is ~190 MB, but use 10 MB as a conservative validity floor.
    """
    adapter_file = path / "adapter_model.safetensors"
    config_file = path / "adapter_config.json"
    return (
        adapter_file.is_file()
        and adapter_file.stat().st_size >= 10_000_000
        and config_file.is_file()
    )


def _ensure_git_lfs_available() -> None:
    if shutil.which("git") is None:
        raise RuntimeError(
            "git is not installed in the image. "
            "The Docker image must include git + git-lfs when adapters are not baked in."
        )
    if shutil.which("git-lfs") is None and shutil.which("git") is not None:
        # Some images expose LFS only as `git lfs`.
        probe = subprocess.run(
            ["git", "lfs", "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "git-lfs is not installed in the image and the adapter files "
                "are not available locally."
            )


def _adapter_diagnostics(repo_dir: Path) -> Dict[str, Any]:
    """Return safe diagnostics without exposing credentials."""
    diagnostics: Dict[str, Any] = {
        "repo_dir": str(repo_dir),
        "files": {},
        "lfs_ls_files_tail": "",
    }

    for rel in (
        f"{SPECIALIST_REL}/adapter_model.safetensors",
        f"{SPECIALIST_REL}/adapter_config.json",
        f"{META_REL}/adapter_model.safetensors",
        f"{META_REL}/adapter_config.json",
    ):
        p = repo_dir / rel
        diagnostics["files"][rel] = {
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
        }

    try:
        diagnostics["lfs_ls_files_tail"] = _run(
            ["git", "lfs", "ls-files"],
            cwd=repo_dir,
        )[-5000:]
    except Exception as exc:
        diagnostics["lfs_ls_files_tail"] = f"unavailable: {type(exc).__name__}: {exc}"

    return diagnostics


def _clone_adapters_from_github() -> Tuple[Path, Path]:
    """
    Clone the private repository without smudging every LFS object, then fetch
    the exact Specialist + Meta adapter weights.

    Important robustness detail:
    some Git-LFS/container combinations report a successful `git lfs pull`
    while the working-tree file remains an LFS pointer. We therefore:
      1) fetch exact adapter_model.safetensors objects;
      2) explicitly run `git lfs checkout` on those paths;
      3) validate real file size;
      4) if still not materialized, fall back to the known-good FULL
         `git lfs pull` flow used successfully in the training notebooks;
      5) validate again and return detailed diagnostics on failure.
    """
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Specialist/Meta adapters are not real local checkpoints and "
            "GITHUB_TOKEN is missing, so Git LFS cannot fetch them."
        )

    _ensure_git_lfs_available()

    repo_dir = Path("/tmp/athar_council_runtime")
    shutil.rmtree(repo_dir, ignore_errors=True)

    clone_url = f"https://x-access-token:{token}@github.com/{GITHUB_REPO}.git"
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"

    _log(
        f"Cloning {GITHUB_REPO}@{GITHUB_BRANCH} without automatic LFS smudge."
    )
    _run(
        [
            "git", "clone",
            "--depth", "1",
            "--branch", GITHUB_BRANCH,
            clone_url,
            str(repo_dir),
        ],
        env=env,
    )

    _run(["git", "lfs", "install", "--local"], cwd=repo_dir, env=env)

    specialist = repo_dir / SPECIALIST_REL
    meta = repo_dir / META_REL

    specialist_weight = f"{SPECIALIST_REL}/adapter_model.safetensors"
    meta_weight = f"{META_REL}/adapter_model.safetensors"
    exact_include = f"{specialist_weight},{meta_weight}"

    # Stage 1: exact LFS fetch + explicit checkout.
    _log("Fetching exact Specialist + Meta LFS adapter weights.")
    targeted_error = None
    try:
        _run(
            [
                "git", "lfs", "fetch",
                "origin", GITHUB_BRANCH,
                f"--include={exact_include}",
                "--exclude=",
            ],
            cwd=repo_dir,
            env=env,
        )
        _run(
            [
                "git", "lfs", "checkout",
                specialist_weight,
                meta_weight,
            ],
            cwd=repo_dir,
            env=env,
        )
    except Exception as exc:
        targeted_error = f"{type(exc).__name__}: {exc}"
        _log(
            "Targeted Git-LFS materialization did not complete; "
            "will try full pull fallback."
        )

    if (
        _real_adapter_checkpoint(specialist)
        and _real_adapter_checkpoint(meta)
    ):
        _log("Specialist + Meta adapters materialized via targeted Git LFS fetch.")
        return specialist, meta

    # Stage 2: known-good fallback. This is the exact broad pattern that
    # previously materialized the four ~190 MB adapters in Kaggle.
    _log(
        "Targeted LFS fetch left pointer/missing files. "
        "Running full `git lfs pull` fallback."
    )
    full_pull_error = None
    try:
        _run(["git", "lfs", "pull"], cwd=repo_dir, env=env)

        # Explicit checkout is harmless if pull already smudged the files and
        # fixes environments where objects were fetched but pointers remained.
        _run(
            [
                "git", "lfs", "checkout",
                specialist_weight,
                meta_weight,
            ],
            cwd=repo_dir,
            env=env,
        )
    except Exception as exc:
        full_pull_error = f"{type(exc).__name__}: {exc}"

    if (
        _real_adapter_checkpoint(specialist)
        and _real_adapter_checkpoint(meta)
    ):
        _log("Specialist + Meta adapters materialized via full Git LFS fallback.")
        return specialist, meta

    diagnostics = _adapter_diagnostics(repo_dir)

    raise RuntimeError(
        "Git LFS finished without materializing the required adapters. "
        f"targeted_error={targeted_error!r}; "
        f"full_pull_error={full_pull_error!r}; "
        f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
    )


def _resolve_adapter_paths() -> Tuple[Path, Path]:
    """
    Prefer weights already baked/mounted in the image. Only use GitHub when
    necessary. Nothing in this function runs during worker boot.
    """
    local_specialist = ROOT / SPECIALIST_REL
    local_meta = ROOT / META_REL

    if (
        _real_adapter_checkpoint(local_specialist)
        and _real_adapter_checkpoint(local_meta)
    ):
        _log("Using local Specialist + Meta adapter checkpoints.")
        return local_specialist, local_meta

    return _clone_adapters_from_github()


# ---------------------------------------------------------------------------
# Lazy model / council bootstrap — runs on first real consultation job only
# ---------------------------------------------------------------------------

def _ensure_engine():
    global _ENGINE, _BASE_MODEL, _TOKENIZER

    if _ENGINE is not None:
        return _ENGINE

    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE

        started = time.time()
        _log("First advisory request: starting lazy Qwen/Council initialization.")

        # Heavy imports deliberately live here, NOT at process startup.
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        # Import the council only after a job is assigned. This guarantees that
        # a typo/import issue in the council cannot make the RunPod worker exit
        # before it registers as a worker.
        from handler_advisory_council import AtharCouncilEngine

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This advisory endpoint requires a GPU worker."
            )

        specialist_path, meta_path = _resolve_adapter_paths()

        _log(f"Loading tokenizer: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # A40 supports BF16; retain a safe FP16 fallback for other GPU types.
        use_bf16 = bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        # SDPA is intentionally the default here. It avoids making worker
        # availability depend on flash-attn being installed/ABI-compatible.
        attn_impl = os.getenv("ATHAR_ATTN_IMPLEMENTATION", "sdpa").strip() or "sdpa"

        _log(
            f"Loading {MODEL_ID} in 4-bit | dtype={compute_dtype} | attention={attn_impl}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
            device_map={"": 0},
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        model.eval()

        engine = AtharCouncilEngine(
            base_model=model,
            tokenizer=tokenizer,
            specialist_adapter_path=specialist_path,
            meta_adapter_path=meta_path,
            model_lock=_ENGINE_LOCK,
        )

        _BASE_MODEL = engine.model
        _TOKENIZER = tokenizer
        _ENGINE = engine

        _log(
            "Athar Advisory Council initialized successfully in "
            f"{round(time.time() - started, 2)}s."
        )
        return _ENGINE


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def _safe_preflight(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPORTANT: this intentionally does not load Qwen or import the council file.
    It lets us prove the worker + RunPod entrypoint are healthy first.
    """
    local_specialist = ROOT / SPECIALIST_REL
    local_meta = ROOT / META_REL

    return {
        "status": "advisory_consultation_preflight_ok",
        "boot_version": BOOT_VERSION,
        "python": sys.version.split()[0],
        "root": str(ROOT),
        "model": MODEL_ID,
        "cuda_model_not_loaded": True,
        "specialist_local_real": _real_adapter_checkpoint(local_specialist),
        "meta_local_real": _real_adapter_checkpoint(local_meta),
        "specialist_local_weight_bytes": (
            (local_specialist / "adapter_model.safetensors").stat().st_size
            if (local_specialist / "adapter_model.safetensors").exists()
            else None
        ),
        "meta_local_weight_bytes": (
            (local_meta / "adapter_model.safetensors").stat().st_size
            if (local_meta / "adapter_model.safetensors").exists()
            else None
        ),
        "github_token_present": bool(os.getenv("GITHUB_TOKEN")),
        "message": (
            "Worker/entrypoint is healthy. Heavy model + council initialization "
            "will happen only on the first real advisory_consultation request."
        ),
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Production-safe Screen-3 handler.

    Expected RunPod envelope:
        {"input": {...}}
    """
    try:
        if not isinstance(job, dict):
            return {"status": "FAILED", "error": "RunPod job must be an object."}

        job_input = job.get("input", {})
        if not isinstance(job_input, dict):
            return {
                "status": "FAILED",
                "error": "RunPod input must be a JSON object.",
            }

        request_type = str(job_input.get("type") or "").strip()

        if request_type != "advisory_consultation":
            return {
                "status": "FAILED",
                "error": (
                    "This startup-safe production entrypoint currently serves "
                    "type='advisory_consultation'. "
                    f"Received type={request_type!r}."
                ),
            }

        if bool(job_input.get("preflight", False)):
            return _safe_preflight(job_input)

        engine = _ensure_engine()
        return engine.consult(job_input)

    except Exception as exc:
        # Crucially: a job failure must NOT terminate the worker process.
        # Return a normal RunPod result and keep the warm worker alive.
        trace = traceback.format_exc()
        _log(f"Job failed but worker will remain alive: {type(exc).__name__}: {exc}")
        print(trace[-12000:], flush=True)

        return {
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Serverless process entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    _log(
        f"Booting {BOOT_VERSION}; heavy ML imports are lazy. "
        f"ROOT={ROOT}"
    )
    # If this line is visible in Container Logs, Python reached the RunPod SDK.
    _log("Registering RunPod serverless handler now.")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
