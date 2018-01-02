from ..core import (
    _direction, _bearing, _mag, _norm, _norm_angle,
    _wrap_unit_square, _collisions,
    find_better, create, DEFAULT_CONFIG, step, swap_ships,
    load_log, save_log,
)
import itertools as it
import numpy as np


def test_direction():
    np.testing.assert_allclose(_direction(0), [0, 1], atol=1e-7)
    np.testing.assert_allclose(
        _direction(np.arange(0, 2 * np.pi, np.pi/2)),
        [[0, 1],
         [1, 0],
         [0, -1],
         [-1, 0]], atol=1e-7)


def test_bearing():
    np.testing.assert_allclose(_bearing(np.array([0, 1])), 0, atol=1e-7)
    np.testing.assert_allclose(
        _bearing(np.array([
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0]])),
        [0, np.pi / 2, np.pi, -np.pi / 2])


def test_mag_norm_angle():
    np.testing.assert_allclose(_mag(np.array([3, 4])), 5)
    np.testing.assert_allclose(_norm(np.array([3, 4])), [0.6, 0.8])
    np.testing.assert_allclose(_norm_angle(2 * np.pi + 0.5), 0.5)
    np.testing.assert_allclose(_norm_angle(-4 * np.pi - 0.5), -0.5)


def test_wrap_unit_square():
    np.testing.assert_allclose(
        _wrap_unit_square(np.array([
            [1.01, -0.95],
            [0.95, -1.01],
        ])),
        np.array([
            [-0.99, -0.95],
            [0.95, 0.99]
        ]))


def test_collisions():
    np.testing.assert_equal(_collisions(np.array(
        [[0, 0],
         [1.9, 1.9],
         [3.8, 1.9],
         [3.8, 0.0]]
    ), np.array([1, 1, 2, 0])), [
        False,
        True,
        True,
        True,
    ])


def test_find_better():
    assert find_better(it.repeat(0), max_trials=100, threshold=0.99) == 0
    assert find_better(it.repeat(1), max_trials=100, threshold=0.99) == 1
    assert find_better(
        it.cycle([0, 1]), max_trials=1000, threshold=0.99) is None


def _check_shape(bodies, n, no_b=False):
    assert bodies.x.shape == (n, 2)
    assert bodies.dx.shape == (n, 2)
    if no_b:
        assert bodies.b is None
    else:
        assert bodies.b.shape == (n,)


def _check_state(state):
    _check_shape(state.ships, 2)
    _check_shape(state.planets, 2, no_b=True)
    _check_shape(state.bullets, 0, no_b=True)


def test_create_step_swap_roundtrip():
    state_0 = create(DEFAULT_CONFIG._replace(max_planets=2))
    _check_state(state_0)

    state_1, reward = step(state_0, np.array([2, 2]), DEFAULT_CONFIG)
    _check_state(state_1)
    np.testing.assert_allclose(reward, 0)

    _check_state(swap_ships(state_1))

    # roundtrip via file
    save_log('/tmp/astro.test.log', DEFAULT_CONFIG, [state_0, state_1])
    re_config, re_states = load_log('/tmp/astro.test.log')
    np.testing.assert_equal(re_config, DEFAULT_CONFIG)
    assert len(re_states) == 2
    np.testing.assert_equal(re_states, [state_0, state_1])
