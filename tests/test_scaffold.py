"""M1 smoke tests.

These do two jobs. The obvious one is proving the package imports and the
config behaves. The less obvious one -- and the reason it is worth writing on
day one -- is enforcing the Repository Standard as executable rules. A README
that says "every project has ADRs" is a wish; a test that fails the build when
the ADR folder is empty is a policy.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from pecos import __version__
from pecos.config import REPO_ROOT, Settings, settings

# ---------------------------------------------------------------------------
# Package basics
# ---------------------------------------------------------------------------


def test_package_imports_and_has_version() -> None:
    """The package is importable and advertises a version string."""
    assert isinstance(__version__, str)
    assert __version__.count(".") == 2, "version should look like MAJOR.MINOR.PATCH"


def test_repo_root_resolves_to_the_actual_repo() -> None:
    """REPO_ROOT must point at the folder containing pyproject.toml.

    This catches the classic breakage where someone moves config.py deeper into
    the package and every relative path in the project silently shifts.
    """
    assert (REPO_ROOT / "pyproject.toml").exists()


# ---------------------------------------------------------------------------
# Config behaviour
# ---------------------------------------------------------------------------


def test_settings_is_frozen() -> None:
    """Settings must be immutable, so eval provenance cannot be falsified."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.seed = 1  # type: ignore[misc]


def test_default_seed_is_deterministic() -> None:
    """Two Settings built in the same environment agree on the seed."""
    assert Settings().seed == Settings().seed


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """PC_-prefixed env vars override the built-in defaults."""
    monkeypatch.setenv("PC_SEED", "999")
    assert Settings().seed == 999


def test_empty_env_var_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty string is treated as unset, not as a value.

    On Windows, clearing a variable with `$env:PC_SEED = ""` leaves it defined
    but empty. Accepting that would produce a confusing crash three modules
    downstream instead of a sane default here.
    """
    monkeypatch.setenv("PC_SEED", "")
    assert Settings().seed == 20260804


def test_garbage_env_var_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric value in a numeric field does not crash the run."""
    monkeypatch.setenv("PC_RETRIEVE_K_DENSE", "not-a-number")
    assert Settings().retrieve_k_dense == 20


def test_ci_mode_shrinks_the_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI mode reduces volume so the suite runs fast."""
    monkeypatch.setenv("PC_CI_MODE", "0")
    full_run = Settings().n_loan_packages
    monkeypatch.setenv("PC_CI_MODE", "1")
    ci_run = Settings()
    assert ci_run.ci_mode is True
    assert ci_run.n_loan_packages < full_run


def test_ci_mode_package_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit counts: 4 packages in CI, 40 in a full run."""
    monkeypatch.setenv("PC_CI_MODE", "1")
    assert Settings().n_loan_packages == 4
    monkeypatch.setenv("PC_CI_MODE", "0")
    assert Settings().n_loan_packages == 40


def test_temperature_defaults_to_zero() -> None:
    """Non-zero temperature when extracting financial figures is error injection."""
    assert settings.temperature == 0.0


def test_postgres_port_avoids_sibling_projects() -> None:
    """P2 uses 5433 and P4 uses 5434; a reviewer may run all three at once."""
    assert settings.pg_port == 5435


def test_dsn_is_well_formed() -> None:
    """The assembled connection string contains host, port, and database."""
    dsn = settings.pg_dsn
    assert dsn.startswith("postgresql://")
    assert ":5435/" in dsn
    assert dsn.endswith("/pecos")


def test_public_dict_masks_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secrets never appear in anything written to a report file."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    public = Settings().public_dict()
    assert public["openai_api_key"] == "SET"
    assert "sk-super-secret-value" not in str(public)
    assert public["pg_password"] in {"SET", "UNSET"}


def test_public_dict_is_json_serialisable() -> None:
    """Paths must be stringified so eval metadata dumps without a custom encoder."""
    import json

    json.dumps(settings.public_dict())


def test_cost_ceiling_is_positive_and_bounded() -> None:
    """An unbounded agent loop is this project's most expensive failure mode."""
    assert 0 < settings.max_cost_usd_per_memo <= 5.0


# ---------------------------------------------------------------------------
# Repository Standard, enforced
# ---------------------------------------------------------------------------

REQUIRED_PATHS = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    ".github/workflows/ci.yml",
    "docs/business-brief.md",
    "docs/architecture.md",
    "docs/data-dictionary.md",
    "docs/runbook.md",
    "docs/decisions",
    "src/pecos/__init__.py",
    "src/pecos/config.py",
    "tests",
    "evals/datasets",
]


@pytest.mark.parametrize("relative_path", REQUIRED_PATHS)
def test_required_repository_artifact_exists(relative_path: str) -> None:
    """Every artefact the Repository Standard requires is present."""
    assert (REPO_ROOT / relative_path).exists(), f"missing: {relative_path}"


def test_at_least_three_adrs_exist() -> None:
    """Design decisions are recorded, not remembered."""
    adrs = list((REPO_ROOT / "docs" / "decisions").glob("[0-9][0-9][0-9][0-9]-*.md"))
    assert len(adrs) >= 3


def test_no_env_file_is_committed() -> None:
    """A real .env in the repo means a leaked key. Fail loudly.

    The Gemini key exposed in a screenshot earlier this year is exactly the
    class of incident this test exists to prevent.
    """
    assert not (REPO_ROOT / ".env").exists() or _is_gitignored(".env")


def _is_gitignored(name: str) -> bool:
    """Crude check that a name appears in .gitignore."""
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    return name in text


def test_gitignore_covers_the_dangerous_paths() -> None:
    """Data, secrets, and model caches must never reach the remote."""
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in [".env", "data/", "__pycache__", ".venv"]:
        assert pattern in text, f".gitignore is missing {pattern}"


def test_env_example_documents_every_pc_variable() -> None:
    """Every PC_ setting the code reads is documented in .env.example.

    Prevents the situation where a reviewer clones the repo and cannot work out
    which variables exist without reading the source.
    """
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    source = (REPO_ROOT / "src" / "pecos" / "config.py").read_text(encoding="utf-8")

    referenced = set()
    for line in source.splitlines():
        if '"PC_' in line:
            start = line.index('"PC_') + 1
            end = line.index('"', start)
            referenced.add(line[start:end])

    missing = sorted(name for name in referenced if name not in example)
    assert not missing, f"undocumented settings: {missing}"


def test_working_directory_independence() -> None:
    """Paths resolve from the module location, not from os.getcwd().

    pytest is often launched from a different directory than the repo root;
    anything that depends on cwd breaks the moment CI runs it.
    """
    original = Path.cwd()
    try:
        os.chdir("/")
        assert (Settings().repo_root / "pyproject.toml").exists()
    finally:
        os.chdir(original)
