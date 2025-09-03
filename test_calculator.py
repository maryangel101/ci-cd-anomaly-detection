import random
from calculator import add, subtract, divide, multiply

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 10) == 0

def test_divide():
    assert divide(10, 2) == 5
    assert divide(9, 3) == 3

# # Flaky test that fails randomly
# def test_flaky():
#     if random.random() < 0.3:  # 30% chance to fail
#         assert False, "Flaky test failed randomly"
#     assert True

# Always failing test (comment out initially)
# def test_always_fail():
#     assert 1 == 2, "This test always fails"