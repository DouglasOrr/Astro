from ..core import (
    _collisions, create, DEFAULT_CONFIG, step, swap_ships,
    load_log, save_log,
)
import numpy as np


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
