from .. import core, script
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


def test_create_step_roll():
    for base in [core.DEFAULT_CONFIG,
                 core.SOLO_CONFIG,
                 core.SOLO_EASY_CONFIG]:
        for config in it.islice(core.generate_configs(base), 10):
            # create
            state_0 = core.create(config)
            np.testing.assert_equal(state_0, core.create(config),
                                    'create should be deterministic')
            _check_state(state_0, config)

            # step
            control = np.full([1 if config.solo else 2], 2)
            state_1, reward = core.step(state_0, control, config)
            _check_state(state_1, config)
            np.testing.assert_allclose(reward, 0)

            # other methods
            if not config.solo:
                _check_state(core.roll_ships(state_1, 1), config)


def test_play_solo_easy():
    for config in it.islice(core.generate_configs(core.SOLO_EASY_CONFIG), 3):
        # should always fall into a planet & lose
        game = core.play(config, [script.NothingBot()])
        assert game.config == config
        assert game.winner is None

        # roundtrip via file
        core.save_log('/tmp/astro.test.log', game)
        np.testing.assert_equal(core.load_log('/tmp/astro.test.log'), game)


def test_play_solo():
    for config in it.islice(core.generate_configs(core.SOLO_CONFIG), 3):
        # might not fall into the planet (e.g starting in the middle)
        game = core.play(config, [script.NothingBot()])
        assert game.config == config

        # roundtrip via file
        core.save_log('/tmp/astro.test.log', game)
        np.testing.assert_equal(core.load_log('/tmp/astro.test.log'), game)


def test_play():
    for config in it.islice(core.generate_configs(core.DEFAULT_CONFIG), 3):
        game = core.play(config, [script.NothingBot(), script.NothingBot()])
        assert game.config == config

        # roundtrip via file
        core.save_log('/tmp/astro.test.log', game)
        np.testing.assert_equal(core.load_log('/tmp/astro.test.log'), game)


def test_script():
    for config in it.islice(core.generate_configs(
            core.SOLO_CONFIG._replace(max_time=20)), 3):
        game = core.play(config, [script.ScriptBot.create(config)])
        assert game.winner == 0

    for config in it.islice(core.generate_configs(
            core.DEFAULT_CONFIG._replace(max_time=20)), 3):
        game = core.play(config, [
            script.NothingBot(), script.ScriptBot.create(config)])
        assert game.winner == 1
