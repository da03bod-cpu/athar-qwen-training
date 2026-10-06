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
import re
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
    "checkpoints/specialist-v2",
).strip("/")
META_REL = os.getenv(
    "COUNCIL_META_REL",
    "checkpoints/meta",
).strip("/")

# Historical repositories used both `checkpoints/specialist` and
# `checkpoints/specialist-v2`. Production must tolerate either layout.
SPECIALIST_REL_CANDIDATES = tuple(dict.fromkeys([
    SPECIALIST_REL,
    "checkpoints/specialist-v2",
    "checkpoints/specialist",
]))
META_REL_CANDIDATES = tuple(dict.fromkeys([
    META_REL,
    "checkpoints/meta",
]))

_ENGINE = None
_ENGINE_LOCK = threading.RLock()

# Keep references to the shared model/tokenizer so they are not garbage
# collected and so every subsequent request reuses the warm worker.
_BASE_MODEL = None
_TOKENIZER = None

BOOT_VERSION = "athar-runpod-safe-entry-v3.1-lfs-autodiscovery"


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
        "candidate_files": {},
        "lfs_ls_files_tail": "",
    }

    rels = list(SPECIALIST_REL_CANDIDATES) + list(META_REL_CANDIDATES)
    for rel in rels:
        weight = repo_dir / rel / "adapter_model.safetensors"
        config = repo_dir / rel / "adapter_config.json"
        diagnostics["candidate_files"][rel] = {
            "weight_exists": weight.exists(),
            "weight_size_bytes": weight.stat().st_size if weight.exists() else None,
            "config_exists": config.exists(),
        }

    try:
        diagnostics["lfs_ls_files_tail"] = _run(
            ["git", "lfs", "ls-files"],
            cwd=repo_dir,
        )[-5000:]
    except Exception as exc:
        diagnostics["lfs_ls_files_tail"] = (
            f"unavailable: {type(exc).__name__}: {exc}"
        )

    return diagnostics


def _discover_lfs_adapter_rel(
    repo_dir: Path,
    *,
    kind: str,
) -> Optional[str]:
    """
    Discover the real tracked adapter directory from `git lfs ls-files`.

    Example production repository currently exposes:
      checkpoints/meta/adapter_model.safetensors
      checkpoints/specialist-v2/adapter_model.safetensors

    Older repositories used:
      checkpoints/specialist/adapter_model.safetensors
    """
    output = _run(["git", "lfs", "ls-files"], cwd=repo_dir)

    tracked_paths = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue

        # `git lfs ls-files` commonly prints:
        #   <oid-prefix> * path
        # or
        #   <oid-prefix> - path
        m = re.match(r"^[0-9a-fA-F]+\s+[*-]\s+(.+)$", line)
        if not m:
            continue

        path = m.group(1).strip().replace("\\\\", "/")
        if path.endswith("/adapter_model.safetensors"):
            tracked_paths.append(path)

    # Exclude recovery/pointer archives. They are historical references and
    # some of their LFS objects no longer exist on the server.
    tracked_paths = [
        p for p in tracked_paths
        if not p.startswith("_recovery/")
        and "/_recovery/" not in p
        and "pointer" not in p.lower()
    ]

    if kind == "meta":
        preferred = [
            p for p in tracked_paths
            if re.search(r"(^|/)meta(/|$)", p, flags=re.I)
        ]
    elif kind == "specialist":
        preferred = [
            p for p in tracked_paths
            if re.search(r"(^|/)specialist(?:-v?\d+)?(/|$)", p, flags=re.I)
            or "/specialist-" in p.lower()
            or "/specialist/" in p.lower()
        ]

        # Prefer V2/current specialist if both old and new paths exist.
        preferred.sort(
            key=lambda p: (
                0 if "specialist-v2" in p.lower() else
                1 if "specialist_v2" in p.lower() else
                2
            )
        )
    else:
        raise ValueError(f"Unknown adapter kind: {kind}")

    if not preferred:
        return None

    return str(Path(preferred[0]).parent).replace("\\\\", "/")


def _first_real_local_adapter(
    root: Path,
    candidates,
) -> Optional[Path]:
    for rel in candidates:
        p = root / rel
        if _real_adapter_checkpoint(p):
            return p
    return None


def _materialize_exact_lfs_file(
    repo_dir: Path,
    env: Dict[str, str],
    rel_file: str,
) -> None:
    """
    Fetch and checkout ONE exact LFS object only.

    Never run a repository-wide `git lfs pull`: the production repo contains
    historical `_recovery/lfs_pointers/*` objects whose remote LFS blobs return
    404 and are unrelated to inference.
    """
    _log(f"Fetching exact Git-LFS object: {rel_file}")

    fetch_error = None
    try:
        _run(
            [
                "git", "lfs", "fetch",
                "origin", GITHUB_BRANCH,
                f"--include={rel_file}",
                "--exclude=",
            ],
            cwd=repo_dir,
            env=env,
        )
    except Exception as exc:
        fetch_error = exc

    if fetch_error is None:
        try:
            _run(
                ["git", "lfs", "checkout", rel_file],
                cwd=repo_dir,
                env=env,
            )
            return
        except Exception as exc:
            fetch_error = exc

    # Some LFS versions behave better with `pull --include=<exact file>`.
    _log(
        f"Exact fetch/checkout fallback for {rel_file}: "
        f"{type(fetch_error).__name__}: {fetch_error}"
    )
    _run(
        [
            "git", "lfs", "pull",
            f"--include={rel_file}",
            "--exclude=",
        ],
        cwd=repo_dir,
        env=env,
    )
    _run(
        ["git", "lfs", "checkout", rel_file],
        cwd=repo_dir,
        env=env,
    )


def _clone_adapters_from_github() -> Tuple[Path, Path]:
    """
    Clone normal Git files, discover the repository's CURRENT adapter paths,
    and materialize only the two LFS weight files needed for inference.

    This intentionally avoids full `git lfs pull` because the current repo
    contains historical `_recovery/lfs_pointers` entries whose LFS objects no
    longer exist remotely.
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

    specialist_rel = _discover_lfs_adapter_rel(
        repo_dir,
        kind="specialist",
    )
    meta_rel = _discover_lfs_adapter_rel(
        repo_dir,
        kind="meta",
    )

    if not specialist_rel:
        raise RuntimeError(
            "Could not discover a Specialist adapter in Git LFS. "
            f"diagnostics={json.dumps(_adapter_diagnostics(repo_dir), ensure_ascii=False)}"
        )
    if not meta_rel:
        raise RuntimeError(
            "Could not discover a Meta adapter in Git LFS. "
            f"diagnostics={json.dumps(_adapter_diagnostics(repo_dir), ensure_ascii=False)}"
        )

    _log(
        f"Discovered current adapter paths: "
        f"specialist={specialist_rel}, meta={meta_rel}"
    )

    specialist_weight = f"{specialist_rel}/adapter_model.safetensors"
    meta_weight = f"{meta_rel}/adapter_model.safetensors"

    _materialize_exact_lfs_file(
        repo_dir,
        env,
        specialist_weight,
    )
    _materialize_exact_lfs_file(
        repo_dir,
        env,
        meta_weight,
    )

    specialist_path = repo_dir / specialist_rel
    meta_path = repo_dir / meta_rel

    specialist_ok = _real_adapter_checkpoint(specialist_path)
    meta_ok = _real_adapter_checkpoint(meta_path)

    if not specialist_ok or not meta_ok:
        diagnostics = _adapter_diagnostics(repo_dir)
        raise RuntimeError(
            "Exact Git-LFS materialization did not produce valid adapter "
            "checkpoints. "
            f"specialist_rel={specialist_rel!r}; "
            f"meta_rel={meta_rel!r}; "
            f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
        )

    _log(
        "Specialist + Meta adapters are materialized and validated: "
        f"specialist={specialist_path}, meta={meta_path}"
    )
    return specialist_path, meta_path


def _resolve_adapter_paths() -> Tuple[Path, Path]:
    """
    Prefer real local checkpoints. Support both historical and current
    Specialist directory names. Only hit GitHub when necessary.
    """
    local_specialist = _first_real_local_adapter(
        ROOT,
        SPECIALIST_REL_CANDIDATES,
    )
    local_meta = _first_real_local_adapter(
        ROOT,
        META_REL_CANDIDATES,
    )

    if local_specialist and local_meta:
        _log(
            "Using local Specialist + Meta adapter checkpoints: "
            f"{local_specialist} | {local_meta}"
        )
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
    Zero-model preflight. No Qwen/council load and no GitHub network call.
    """
    specialist_candidates = []
    for rel in SPECIALIST_REL_CANDIDATES:
        p = ROOT / rel
        weight = p / "adapter_model.safetensors"
        specialist_candidates.append({
            "rel": rel,
            "real": _real_adapter_checkpoint(p),
            "weight_bytes": weight.stat().st_size if weight.exists() else None,
        })

    meta_candidates = []
    for rel in META_REL_CANDIDATES:
        p = ROOT / rel
        weight = p / "adapter_model.safetensors"
        meta_candidates.append({
            "rel": rel,
            "real": _real_adapter_checkpoint(p),
            "weight_bytes": weight.stat().st_size if weight.exists() else None,
        })

    return {
        "status": "advisory_consultation_preflight_ok",
        "boot_version": BOOT_VERSION,
        "python": sys.version.split()[0],
        "root": str(ROOT),
        "model": MODEL_ID,
        "cuda_model_not_loaded": True,
        "github_repo": GITHUB_REPO,
        "github_branch": GITHUB_BRANCH,
        "github_token_present": bool(os.getenv("GITHUB_TOKEN")),
        "specialist_candidates": specialist_candidates,
        "meta_candidates": meta_candidates,
        "message": (
            "Worker/entrypoint is healthy. Adapter path discovery and heavy "
            "model initialization happen only on the first real consultation."
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
