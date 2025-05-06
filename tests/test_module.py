import pytest
import numpy as np

@pytest.fixture
def probabilities():
    return np.load("tests/fixtures/probabilities.npy")

def test_challenge_passed(probabilities):
    answers = [
      1,0,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,1,1,0,0,
      1,1,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,1,1,
      0,0,0,0,0,0
    ]
    correct = sum(int(round(p)) == a for p, a in zip(probabilities, answers))
    pct = (correct / len(answers)) * 100
    assert pct >= 63, f"Solo {pct:.2f}% ≥ 63%"