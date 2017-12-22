'''Astro - simple tiny physics simulation for trying out reinforcement
learning.
'''

import numpy as np
import collections
import json
import importlib


Bodies = collections.namedtuple(
    'Bodies',
    ('x', 'dx', 'b', 'ttl'))

State = collections.namedtuple(
    'State',
    ('ships', 'planets', 'bullets', 't'))

Config = collections.namedtuple(
    'Config', (
        'seed',
        'gravity',
        'dt',
        'max_time',
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


def _to_json(obj):
    if hasattr(obj, '_asdict'):
        d = {k: _to_json(v)
             for k, v in obj._asdict().items()}
        _type = type(obj)
        d['_type'] = '{}:{}'.format(_type.__module__, _type.__name__)
        return d
    elif isinstance(obj, np.ndarray):
        return {'_values': obj.tolist(),
                '_shape': obj.shape}
    else:
        return obj


def _from_json(obj):
    if isinstance(obj, dict):
        if obj.keys() == {'_values', '_shape'}:
            a = np.array(obj['_values']).reshape(obj['_shape'])
            return a.astype(np.float32) if a.dtype is np.float64 else a
        _module, _name = obj.pop('_type').split(':')
        return getattr(
            importlib.import_module(_module),
            _name
        )(**{k: _from_json(v) for k, v in obj.items()})
    else:
        return obj


DEFAULT_CONFIG = Config(
    seed=42,
    gravity=0.1,
    dt=0.02,
    max_time=60,
    bullet_spawn=0.5,
    bullet_ttl=1.0,
    bullet_speed=1.0,
    ship_thrust=1.0,
    ship_rspeed=3.0,
    ship_position=np.array([0.9, 0.9], dtype=np.float32),
    ship_radius=0.02,
    planet_orbit=0.5,
    planet_mass=1.0,
    planet_radius=0.2,
)


def _direction(bearing):
    '''Return a unit vector pointing in the direction 'bearing' from +y.
    '''
    return np.stack((np.sin(bearing, dtype=np.float32),
                     np.cos(bearing, dtype=np.float32)),
                    axis=-1)


def _bearing(x):
    '''Return the bearing of the vector x.

    x -- array(... x 2) -- direction vectors

    returns -- array(...) -- bearings from +y
    '''
    return np.arctan2(x[..., 0], x[..., 1])


def _mag(x):
    '''Compute the magnitude of x along the last dimension.

    x -- array(... x 2)

    returns -- array(...)
    '''
    return np.sqrt((x ** 2).sum(axis=-1))


def _norm(x):
    '''Normalize x along the last dimension.

    x -- array(... x 2)

    returns -- array(... x 2)
    '''
    return x / _mag(x)


def _norm_angle(b):
    '''Normalize an angle (/bearing) to the range [-pi, +pi].

    b -- float -- bearing

    returns -- float
    '''
    return ((b + np.pi) % (2 * np.pi)) - np.pi


def create(config):
    '''Create a new game state randomly.

    Currently only supports the "binary stars" map - two equal mass & size
    stars in orbit of one another.
    '''
    random = np.random.RandomState(config.seed)

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
        ships=Bodies(x=np.stack((ship_0, ship_1), axis=0),
                     dx=np.zeros((2, 2), dtype=np.float32),
                     b=(2 * np.pi * random.rand(2).astype(np.float32)),
                     ttl=None),
        planets=Bodies(x=np.stack((planet_0, planet_1), axis=0),
                       dx=np.stack((planet_0_dx, planet_1_dx), axis=0),
                       b=None,
                       ttl=None),
        bullets=Bodies(x=np.zeros((0, 2), dtype=np.float32),
                       dx=np.zeros((0, 2), dtype=np.float32),
                       b=None,
                       ttl=np.zeros(0, dtype=np.float32)),
        t=0.0,
    )


def _gravity(planets, x, config):
    '''Compute the acceleration due to gravity of 'planets' on objects at position
    'x'.

    planets -- astro.Bodies -- only massive bodies

    x -- array(N x 2) -- positions to evaluate gravitational field

    config -- astro.Config -- constants

    returns -- array(N x 2) -- gravitational field (force per unit mass)
    '''
    rx = planets.x[np.newaxis, :, :] - x[:, np.newaxis, :]
    f_rx = (config.gravity * config.planet_mass /
            np.maximum(1e-12, (rx ** 2).sum(axis=2)))
    return (f_rx[:, :, np.newaxis] * rx).sum(axis=1)


def _wrap_unit_square(x):
    '''Wraps x around the unit square.
    '''
    return ((x + 1) % 2) - 1


def _update_bodies(bodies, a, db, dt):
    '''Compute the movement update for 'bodies'.

    bodies -- astro.Bodies -- to update

    a -- array(N x 2) -- acceleration

    db -- array(N) -- change in bearing

    dt -- float -- timestep

    returns -- astro.Bodies
    '''
    # the approximation (dx + dx') / 2 for updating position seems to lead
    # to instability, so just using dx' here
    dx = bodies.dx + a * dt
    return Bodies(
        x=_wrap_unit_square(bodies.x + dt * dx),
        dx=dx,
        b=None if bodies.b is None else bodies.b + db,
        ttl=None if bodies.ttl is None else bodies.ttl - dt)


def _collisions(x, r):
    '''Compute a collision mask between the objects at positions 'xs', with
    radiuses 'rs'.

    x -- array(N x 2) -- positions

    r -- array(N) -- radiuses

    returns -- array(N; bool) -- collisions
    '''
    rx2 = ((x[np.newaxis, :, :] - x[:, np.newaxis, :]) ** 2).sum(axis=2)
    r2 = (r[np.newaxis, :] + r[:, np.newaxis]) ** 2
    return ((rx2 < r2) & ~np.eye(x.shape[0], dtype=np.bool)).any(axis=1)


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

    nships = state.ships.x.shape[0]
    nplanets = state.planets.x.shape[0]
    collisions = _collisions(
        np.concatenate((
            state.ships.x,
            state.planets.x,
            state.bullets.x), axis=0),
        np.concatenate((
            np.repeat(config.ship_radius, nships),
            np.repeat(config.planet_radius, nplanets),
            np.zeros(state.bullets.x.shape[0])), axis=0))

    if collisions[:nships].any():
        # End of game (collision)
        return None, (1 - 2 * collisions[:nships])

    if config.max_time <= state.t + config.dt:
        # End of game (timeout) - return reward & terminating state
        return None, np.zeros(nships, dtype=np.float32)

    next_state = State(
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
        t=state.t + config.dt,
    )
    return next_state, np.zeros(nships, dtype=np.float32)


def swap_ships(state):
    '''Swap the two ships in "state", so that bots can play against each other.
    '''
    s = state.ships
    return State(
        ships=Bodies(x=s.x[::-1], dx=s.dx[::-1], b=s.b[::-1], ttl=None),
        planets=state.planets,
        bullets=state.bullets,
        t=state.t)


class SurvivalBot:
    '''A simple "staying alive" scripted bot, which tries not to crash
    into the planets

    state -- astro.State

    returns -- int -- control
    '''
    DEFAULT_ARGS = dict(
        max_speed=0.5,
        angle_threshold=0.5,
    )

    def __init__(self, args):
        self.args = args

    def __call__(self, state):
        speed = _mag(state.ships.dx[0])
        if self.args['max_speed'] < speed:
            # slow down!
            target_b = _bearing(-state.ships.dx[0])
        else:
            # avoid the planet
            planets = state.ships.x[0][np.newaxis, :] - state.planets.x
            target_b = _bearing(planets[np.argmin((planets ** 2).sum(axis=1))])
        angle = _norm_angle(target_b - state.ships.b[0])
        if angle < -self.args['angle_threshold']:
            return 0  # rotate left
        elif self.args['angle_threshold'] < angle:
            return 4  # rotate right
        elif np.dot(state.ships.dx[0], _direction(state.ships.b[0])) < 0:
            return 3  # forward
        else:
            return 2  # nothing


def save_log(path, config, states):
    '''Saves a game log as jsonlines.
    '''
    with open(path, 'w') as f:
        f.write(json.dumps(_to_json(config)) + '\n')
        for state in states:
            f.write(json.dumps(_to_json(state)) + '\n')


def load_log(path):
    '''Load a game log from file.

    path -- string -- file path

    returns -- (astro.Config, [astro.State])
    '''
    with open(path, 'r') as f:
        config = _from_json(json.loads(next(f)))
        states = [_from_json(json.loads(line)) for line in f]
        return config, states


# Tests

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


def test_create_step_swap_roundtrip():
    state_0 = create(DEFAULT_CONFIG)
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
