'''Astro - simple tiny physics simulation for trying out reinforcement
learning.
'''

import collections
import numpy as np


Bodies = collections.namedtuple(
    'Bodies',
    ('x', 'dx', 'b', 'ttl'))

State = collections.namedtuple(
    'State',
    ('ships', 'planets', 'bullets'))

Config = collections.namedtuple(
    'Config', (
        'gravity',
        'dt',
        'bullet_spawn',
        'bullet_ttl',
        'bullet_speed',
        'ship_thrust',
        'ship_rspeed',
        'ship_position',
        'ship_radius',
        'planet_orbit',
        'planet_mass',
        'planet_radius',
    ))


DEFAULT_CONFIG = Config(
    gravity=0.8,
    dt=0.1,
    bullet_spawn=0.5,
    bullet_ttl=1.0,
    bullet_speed=1,
    ship_thrust=0.5,
    ship_rspeed=2.0,
    ship_position=np.array([0.9, 0.9], dtype=np.float32),
    ship_radius=0.01,
    planet_orbit=0.5,
    planet_mass=1.0,
    planet_radius=0.1,
)


def _direction(bearing):
    '''Return a unit vector pointing in the direction 'bearing' from +y.
    '''
    return np.stack((np.sin(bearing, dtype=np.float32),
                     np.cos(bearing, dtype=np.float32)),
                    axis=-1)


def create(config, seed):
    '''Create a new game state randomly.

    Currently only supports the "binary stars" map - two equal mass & size
    stars in orbit of one another.

    seed -- int -- random seed
    '''
    random = np.random.RandomState(seed)

    # ships
    ship_0 = (config.ship_position *
              np.sign(random.rand(2).astype(np.float32) - 0.5))
    ship_1 = -ship_0

    # planets
    orientation = 2 * np.pi * random.rand()
    planet_0 = config.planet_orbit * _direction(orientation)
    planet_0_dx = (np.sqrt(config.gravity * config.planet_mass / 2) *
                   _direction(orientation + np.pi / 2))
    planet_1 = -planet_0
    planet_1_dx = -planet_0_dx

    return State(
        Bodies(x=np.stack((ship_0, ship_1), axis=0),
               dx=np.zeros((2, 2), dtype=np.float32),
               b=(2 * np.pi * random.rand(2).astype(np.float32)),
               ttl=None),
        Bodies(x=np.stack((planet_0, planet_1), axis=0),
               dx=np.stack((planet_0_dx, planet_1_dx), axis=0),
               b=None,
               ttl=None),
        Bodies(x=np.zeros((0, 2), dtype=np.float32),
               dx=np.zeros((0, 2), dtype=np.float32),
               b=None,
               ttl=np.zeros(0, dtype=np.float32)),
    )


def _gravity(planets, x, config):
    rx = planets.x[np.newaxis, :, :] - x[:, np.newaxis, :]
    f_rx = (config.gravity * config.planet_mass /
            np.maximum(1e-12, (rx ** 2).sum(axis=2)))
    return (f_rx[:, :, np.newaxis] * rx).sum(axis=1)


def _update_bodies(bodies, a, db, dt):
    # the approximation (dx + dx') / 2 for updating position seems to lead
    # to instability, so just using dx' here
    dx = bodies.dx + a * dt
    return Bodies(
        x=bodies.x + dt * dx,
        dx=dx,
        b=None if bodies.b is None else bodies.b + db,
        ttl=None if bodies.ttl is None else bodies.ttl - dt)


def step(state, control, config):
    '''Advance the world state by a single tick of the game clock.

    state -- astro.State -- old state of the world

    control -- array([2; int]) -- control input for each ship, as follows:

      0 - left
      1 - left + forward
      2 - none
      3 - forward
      4 - right
      5 - right + forward

    config -- astro.Config -- constants defining the world

    returns -- (astro.State or None, array([2; float])) -- next state and
               reward for each ship
    '''
    ships_a = (config.ship_thrust *
               (control % 2)[:, np.newaxis] *
               _direction(state.ships.b) +
               _gravity(state.planets, state.ships.x, config=config))

    ships_db = config.dt * config.ship_rspeed * ((control // 2) - 1)

    return State(
        ships=_update_bodies(
            state.ships,
            a=ships_a,
            db=ships_db,
            dt=config.dt),
        planets=_update_bodies(
            state.planets,
            a=_gravity(state.planets, state.planets.x, config=config),
            db=0,
            dt=config.dt),
        bullets=_update_bodies(
            state.bullets,
            a=0,
            db=0,
            dt=config.dt),
    )


# Tests

def test_direction():
    np.testing.assert_allclose(_direction(0), [0, 1], atol=1e-7)
    np.testing.assert_allclose(
        _direction(np.arange(0, 2 * np.pi, np.pi/2)),
        [[0, 1],
         [1, 0],
         [0, -1],
         [-1, 0]], atol=1e-7)


def _check_shape(bodies, n, no_b=False, no_ttl=False):
    assert bodies.x.shape == (n, 2)
    assert bodies.dx.shape == (n, 2)
    if no_b:
        assert bodies.b is None
    else:
        assert bodies.b.shape == (n,)
    if no_ttl:
        assert bodies.ttl is None
    else:
        assert bodies.ttl.shape == (n,)


def _check_state(state):
    _check_shape(state.ships, 2, no_ttl=True)
    _check_shape(state.planets, 2, no_b=True, no_ttl=True)
    _check_shape(state.bullets, 0, no_b=True)


def test_create_step():
    state = create(DEFAULT_CONFIG, 100)
    _check_state(state)
    state = step(state, np.array([2, 2]), DEFAULT_CONFIG)
    _check_state(state)
