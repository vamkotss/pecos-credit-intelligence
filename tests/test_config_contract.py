"""M2's contract with M1's config module.

The corpus generator is deliberately decoupled from `settings` -- it takes an
explicit `CorpusSpec`. But `scripts/generate_corpus.py` still has to get a seed
and a CI flag from somewhere, and that somewhere is `pecos.config`.

This test pins that dependency to two attribute names. If `Settings` is ever
refactored, this fails with a sentence explaining what to change, instead of an
AttributeError surfacing halfway through a corpus build.
"""

from __future__ import annotations

from pecos.config import REPO_ROOT, settings

REQUIRED = ("seed", "ci_mode")


def test_settings_exposes_what_the_corpus_generator_needs():
    missing = [name for name in REQUIRED if not hasattr(settings, name)]
    assert not missing, (
        f"pecos.config.settings is missing {missing}. The corpus generator reads "
        f"only these two. Add them to Settings in src/pecos/config.py, or update "
        f"scripts/generate_corpus.py to read whatever they were renamed to."
    )


def test_seed_is_an_integer():
    """A seed read as a string would silently produce a different corpus than a
    seed read as an integer, and the manifest hash would change with no code
    change to explain it."""
    assert isinstance(settings.seed, int)


def test_repo_root_resolves_to_this_repository():
    assert (REPO_ROOT / "src" / "pecos").is_dir()
    assert (REPO_ROOT / "tests").is_dir()
