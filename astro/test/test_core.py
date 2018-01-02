from .. import core
import itertools as it
import numpy as np


def test_collisions():
    np.testing.assert_equal(core._collisions(np.array(
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


def _check_state(state, config):
    assert state.ships.x.shape == (1 if config.solo else 2, 2)
    assert state.ships.dx.shape == state.ships.x.shape
    assert state.ships.b.shape == state.ships.x.shape[:1]

    assert 1 <= state.planets.x.shape[0] <= config.max_planets
    assert state.planets.dx.shape == state.planets.x.shape
    assert state.planets.b is None

    assert state.bullets.dx.shape == state.bullets.x.shape
    assert state.bullets.b is None


def test_create_step_swap_roundtrip():
    for base in [core.DEFAULT_CONFIG,
                 core.SOLO_CONFIG,
                 core.SOLO_EASY_CONFIG]:
        for config in it.islice(core.generate_configs(base), 10):
            # create
            state_0 = core.create(config)
            _check_state(state_0, config)

            # step
            control = np.full([1 if config.solo else 2], 2)
            state_1, reward = core.step(state_0, control, config)
            _check_state(state_1, config)
            np.testing.assert_allclose(reward, 0)

            # other methods
            if not config.solo:
                _check_state(core.swap_ships(state_1), config)

            # roundtrip via file
            core.save_log('/tmp/astro.test.log', config, [state_0, state_1])
            re_config, re_states = core.load_log('/tmp/astro.test.log')
            np.testing.assert_equal(re_config, config)
            assert len(re_states) == 2
            np.testing.assert_equal(re_states, [state_0, state_1])
