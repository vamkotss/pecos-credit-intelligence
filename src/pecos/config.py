"""Central configuration for the Pecos credit-intelligence pipeline.

Design rule for this repo: **nothing reads an environment variable directly
except this module.** Every other module imports `settings` from here.

Why that matters for a reviewer: an LLM pipeline has an unusual number of
knobs (model names, temperatures, chunk sizes, retrieval depths, cost caps).
When those are scattered as `os.getenv` calls across twenty files, two things
break. First, you cannot answer "what config produced this eval score?" --
which makes every evaluation result unreproducible. Second, CI cannot run a
cheap, deterministic subset without hunting down each knob.

Both problems are solved by one typed settings object that can serialise
itself into an eval run's metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = three levels up from this file (src/pecos/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_str(key: str, default: str) -> str:
    """Read a string env var, falling back to a default.

    Empty strings are treated as "unset" -- on Windows a variable cleared with
    `$env:PC_FOO = ""` still exists but is empty, and silently accepting that
    would produce confusing downstream failures.
    """
    value = os.environ.get(key, "")
    return value if value.strip() else default


def _env_int(key: str, default: int) -> int:
    """Read an integer env var, falling back to a default on absent/garbage input."""
    raw = _env_str(key, "")
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Read a float env var, falling back to a default on absent/garbage input."""
    raw = _env_str(key, "")
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    """Read a boolean env var. Accepts 1/true/yes/on in any case."""
    raw = _env_str(key, "").lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable, fully-resolved run configuration.

    Frozen on purpose: a settings object that mutates mid-run makes eval
    provenance a lie. If something needs different settings, it builds a new
    Settings, and that new object is what gets recorded.
    """

    # --- Reproducibility -------------------------------------------------
    seed: int = field(default_factory=lambda: _env_int("PC_SEED", 20260804))

    # --- CI mode ---------------------------------------------------------
    # CI mode shrinks the corpus and the eval set so the full pipeline runs in
    # roughly a minute instead of roughly an hour, WITHOUT removing any defect
    # class. Same lesson learned on P2 and P4: keep the hard cases, cut volume.
    ci_mode: bool = field(default_factory=lambda: _env_bool("PC_CI_MODE", False))

    # --- Paths -----------------------------------------------------------
    repo_root: Path = REPO_ROOT
    data_dir: Path = field(
        default_factory=lambda: Path(_env_str("PC_DATA_DIR", str(REPO_ROOT / "data")))
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(
            _env_str("PC_REPORTS_DIR", str(REPO_ROOT / "reports"))
        )
    )
    evals_dir: Path = field(
        default_factory=lambda: Path(_env_str("PC_EVALS_DIR", str(REPO_ROOT / "evals")))
    )

    # --- Vector / relational store ---------------------------------------
    # Port 5435 deliberately: P2 (bluebonnet) uses 5433 and P4 (lonestar) uses
    # 5434. Three portfolio projects must be runnable on one machine without a
    # port collision -- a reviewer who clones two repos and hits "port already
    # allocated" stops reviewing.
    pg_host: str = field(default_factory=lambda: _env_str("PC_PG_HOST", "localhost"))
    pg_port: int = field(default_factory=lambda: _env_int("PC_PG_PORT", 5435))
    pg_db: str = field(default_factory=lambda: _env_str("PC_PG_DB", "pecos"))
    pg_user: str = field(default_factory=lambda: _env_str("PC_PG_USER", "pecos"))
    pg_password: str = field(
        default_factory=lambda: _env_str("PC_PG_PASSWORD", "pecos_local_dev")
    )

    # --- Models ----------------------------------------------------------
    # Two tiers on purpose. Cost engineering (M9) routes cheap work to the
    # small model and only escalates when a router says the task needs it.
    llm_model_small: str = field(
        default_factory=lambda: _env_str("PC_LLM_MODEL_SMALL", "gpt-4o-mini")
    )
    llm_model_large: str = field(
        default_factory=lambda: _env_str("PC_LLM_MODEL_LARGE", "gpt-4o")
    )
    embed_model: str = field(
        default_factory=lambda: _env_str("PC_EMBED_MODEL", "text-embedding-3-small")
    )
    rerank_model: str = field(
        default_factory=lambda: _env_str(
            "PC_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    # Temperature 0 everywhere by default. Non-zero temperature in a system
    # that extracts financial figures is not creativity, it is error injection.
    temperature: float = field(
        default_factory=lambda: _env_float("PC_TEMPERATURE", 0.0)
    )

    # --- Retrieval defaults ----------------------------------------------
    chunk_target_tokens: int = field(
        default_factory=lambda: _env_int("PC_CHUNK_TARGET_TOKENS", 512)
    )
    chunk_overlap_tokens: int = field(
        default_factory=lambda: _env_int("PC_CHUNK_OVERLAP_TOKENS", 64)
    )
    retrieve_k_dense: int = field(
        default_factory=lambda: _env_int("PC_RETRIEVE_K_DENSE", 20)
    )
    retrieve_k_sparse: int = field(
        default_factory=lambda: _env_int("PC_RETRIEVE_K_SPARSE", 20)
    )
    rerank_top_n: int = field(default_factory=lambda: _env_int("PC_RERANK_TOP_N", 6))

    # --- Cost guardrail ---------------------------------------------------
    # A hard ceiling per memo. The agent aborts rather than silently spending.
    # This exists because an unbounded agent loop is the single most expensive
    # failure mode in this project category.
    max_cost_usd_per_memo: float = field(
        default_factory=lambda: _env_float("PC_MAX_COST_USD_PER_MEMO", 0.50)
    )

    # --- Secrets ----------------------------------------------------------
    # Read but never logged, never written into a report, never committed.
    openai_api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY", ""))

    @property
    def pg_dsn(self) -> str:
        """Postgres connection string assembled from the parts above."""
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )

    @property
    def n_loan_packages(self) -> int:
        """How many synthetic loan packages the corpus generator produces.

        CI mode keeps every defect class but only a handful of packages.
        """
        return 4 if self.ci_mode else 40

    def public_dict(self) -> dict[str, str]:
        """Config safe to write into an eval report or model card.

        Secrets are reduced to a presence flag. Paths are stringified so the
        result is JSON-serialisable without a custom encoder.
        """
        out: dict[str, str] = {}
        for key, value in self.__dict__.items():
            if "key" in key or "password" in key:
                out[key] = "SET" if value else "UNSET"
            else:
                out[key] = str(value)
        return out


# The single shared instance. Import this, not the class.
settings = Settings()
