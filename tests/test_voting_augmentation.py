"""Tests for the voting and augmentation modules."""

from src.voting import majority_vote, top_k_by_votes, vote_confidence
from src.augmentation import (
    AUGMENTATIONS,
    augment_train_pairs,
    augment_test_inputs,
)


def test_majority_vote_simple():
    """Majority vote returns the most common grid."""
    g1 = [[1, 2], [3, 4]]
    g2 = [[5, 6], [7, 8]]
    # 3 votes for g1, 1 vote for g2
    result = majority_vote([g1, g1, g2, g1])
    assert result == g1


def test_majority_vote_single():
    g = [[1, 0], [0, 1]]
    assert majority_vote([g]) == g


def test_top_k_returns_two_distinct():
    g1 = [[1, 1], [1, 1]]
    g2 = [[2, 2], [2, 2]]
    g3 = [[3, 3], [3, 3]]
    # g1 appears 3x, g2 appears 2x, g3 appears 1x
    results = top_k_by_votes([g1, g2, g1, g3, g1, g2], k=2)
    assert len(results) == 2
    assert results[0] == g1  # most votes
    assert results[1] == g2  # second most


def test_vote_confidence():
    g1 = [[1, 2], [3, 4]]
    g2 = [[5, 6], [7, 8]]
    winner, conf = vote_confidence([g1, g1, g1, g2])
    assert winner == g1
    assert conf == 0.75  # 3/4


def test_augmentations_are_invertible():
    """Every augmentation's inverse perfectly undoes the forward transform."""
    grid = [[1, 2, 3], [4, 5, 6]]
    for name, fwd, inv in AUGMENTATIONS:
        recovered = inv(fwd(grid))
        assert recovered == grid, f"Augmentation '{name}' is not invertible"


def test_augment_train_pairs():
    """augment_train_pairs transforms both input and output."""
    pair = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    for name, fwd, inv in AUGMENTATIONS:
        aug = augment_train_pairs([pair], fwd)
        assert len(aug) == 1
        aug_inp, aug_out = aug[0]
        # Un-transforming should recover originals
        assert inv(aug_inp) == pair[0], f"Failed for {name}"
        assert inv(aug_out) == pair[1], f"Failed for {name}"


def test_augment_test_inputs():
    tests = [[[1, 2], [3, 4]]]
    for name, fwd, inv in AUGMENTATIONS:
        aug = augment_test_inputs(tests, fwd)
        assert inv(aug[0]) == tests[0], f"Failed for {name}"
