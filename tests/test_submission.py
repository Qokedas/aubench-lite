from __future__ import annotations

from pathlib import Path

import pytest

from cam1k_eval.formats import FormatError
from cam1k_eval.submission import freeze_submission, validate_submission


def test_freeze_preserves_exact_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    before = validate_submission(source)
    frozen = tmp_path / "frozen"
    receipt = freeze_submission(source, frozen)
    assert receipt["sha256"] == before["sha256"]
    assert validate_submission(frozen)["sha256"] == before["sha256"]
    assert (tmp_path / "frozen.freeze.json").is_file()


def test_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "solution.py").write_text("pass\n", encoding="utf-8")
    (source / "escape").symlink_to(tmp_path)
    with pytest.raises(FormatError, match="symlink"):
        validate_submission(source)
