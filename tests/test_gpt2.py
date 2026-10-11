"""
Unit and Synthetic Regression Tests for GPT-2 Small Circuit Analysis Pipeline.
Verifies mathematical correctness of:
1. Low-rank OV operator eigenvalue-based copying score
2. Head behavior detectors on synthetic activation caches
3. Induction stripe diagonal offset indexing
"""

import sys
from pathlib import Path
import torch as t
import pytest

# Add parent directory to path so modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from importlib.machinery import SourceFileLoader

# Load modules by relative path
root = Path(__file__).resolve().parent.parent
mod_10b = SourceFileLoader("mod_10b", str(root / "10b_ov_eigenvalue_copying_score.py")).load_module()
mod_04 = SourceFileLoader("mod_04", str(root / "04_head_behavior_detectors.py")).load_module()
mod_06 = SourceFileLoader("mod_06", str(root / "06_induction_head_score_scan.py")).load_module()


def test_copying_eigenvalue_extremes():
    """Verify that pure positive, pure negative, and balanced operators yield exact scores."""
    copying_score_from_operator = mod_10b.copying_score_from_operator

    pos_mat = t.diag(t.tensor([1.0, 2.0, 3.0]))
    assert abs(copying_score_from_operator(pos_mat) - 1.0) < 1e-6

    neg_mat = t.diag(t.tensor([-1.0, -2.0, -3.0]))
    assert abs(copying_score_from_operator(neg_mat) - (-1.0)) < 1e-6

    balanced_mat = t.diag(t.tensor([1.0, -1.0, 2.0, -2.0]))
    assert abs(copying_score_from_operator(balanced_mat)) < 1e-6


def test_synthetic_head_behavior_detectors():
    """Verify that synthetic patterns trigger the expected head detectors."""
    synthetic_cache = mod_04.make_synthetic_cache()
    current = mod_04.current_attn_detector(synthetic_cache, n_layers=1, n_heads=3)
    prev = mod_04.prev_attn_detector(synthetic_cache, n_layers=1, n_heads=3)
    first = mod_04.first_attn_detector(synthetic_cache, n_layers=1, n_heads=3)

    assert current == ["0.0"]
    assert prev == ["0.1"]
    assert first == ["0.2"]


def test_synthetic_induction_score():
    """Verify that synthetic attention pattern extracts the exact induction stripe."""
    seq_len = 5
    total_len = 1 + 2 * seq_len
    fake_pattern = t.zeros(1, total_len, total_len)
    # Put 1.0 on the induction stripe offset: -(seq_len - 1) = -4
    fake_pattern[0].diagonal(-4).fill_(1.0)

    score = mod_06.induction_score(fake_pattern, seq_len)
    assert t.allclose(score, t.tensor([1.0]))
