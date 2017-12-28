'''Astro - simple tiny physics simulation for trying out reinforcement
learning.
'''

import numpy as np
import scipy as sp
import chainer as C
import scipy.stats  # NOQA
import itertools as it
import collections
import json
import importlib
import sys


Bodies = collections.namedtuple(
    'Bodies',
    ('x', 'dx', 'b'))

State = collections.namedtuple(
    'State',
    ('ships', 'planets', 'bullets',
     'reload', 't'))

Config = collections.namedtuple(
    'Config', (
        # world
        'gravity',
        'dt',
        'max_time',
        'reload_time',
        'bullet_speed',
        'ship_thrust',
        'ship_rspeed',
        'ship_radius',

        # creation
        'seed',
        'outer_ship_position',
        'inner_ship_position',
        'max_planets',
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
    # world
    gravity=0.05,
    dt=0.02,
    max_time=60,
    reload_time=0.3,
    bullet_speed=1.5,
    ship_thrust=1.0,
    ship_rspeed=4.0,
    ship_radius=0.025,

    # creation
    seed=42,
    inner_ship_position=0.2,
    outer_ship_position=0.9,
    max_planets=4,
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
    return x / (_mag(x) + 1e-12)


def _norm_angle(b):
    '''Normalize an angle (/bearing) to the range [-pi, +pi].

    b -- float -- bearing

    returns -- float
    '''
    return ((b + np.pi) % (2 * np.pi)) - np.pi


def _dot(a, b):
    '''Return the dot product between batches of vectors.

    a, b -- array(... x N)

    returns -- array(...)
    '''
    return (a * b).sum(axis=-1)


def _pbetter(nwins, nlosses):
    '''Compute the probability of this being better, given the number
    of wins & losses, under an informative beta prior.

    nwins -- int -- number of wins

    nlosses -- int -- number of losses

    returns -- float -- probability of the underlying win rate being
               greater than 0.5
    '''
    return 1 - sp.stats.beta.cdf(0.5, 1 + nwins, 1 + nlosses)


def create(config):
    '''Create a new game state randomly.

    Currently only supports the "binary stars" map - two equal mass & size
    stars in orbit of one another.
    '''
    random = np.random.RandomState(config.seed)

    # ships
    ship_0 = (config.outer_ship_position *
              np.sign(random.rand(2).astype(np.float32) - 0.5))
    ship_1 = (config.inner_ship_position *
              _direction(2 * np.pi * np.random.rand()))
    if random.rand() < 0.5:
        ship_0, ship_1 = ship_1, ship_0

    # planets
    nplanets = random.randint(2, config.max_planets + 1)
    orientation = (2 * np.pi * random.rand() +
                   np.linspace(0, 2 * np.pi, num=nplanets, endpoint=False))
    reverse = random.choice((-1, 1))
    planets = config.planet_orbit * _direction(orientation)
    planets_dx = (
        np.sqrt(config.gravity * config.planet_mass * (nplanets - 1) / 2) *
        _direction(orientation + reverse * np.pi / 2))

    return State(
        ships=Bodies(x=np.stack((ship_0, ship_1), axis=0),
                     dx=np.zeros((2, 2), dtype=np.float32),
                     b=(2 * np.pi * random.rand(2).astype(np.float32))),
        planets=Bodies(x=planets,
                       dx=planets_dx,
                       b=None),
        bullets=Bodies(x=np.zeros((0, 2), dtype=np.float32),
                       dx=np.zeros((0, 2), dtype=np.float32),
                       b=None),
        reload=0.0,
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


def _mask(bodies, mask):
    '''Only select bodies for which mask is true.

    bodies -- astro.Bodies

    mask -- np.array(N; bool) -- selects bodies to keep

    returns -- astro.Bodies
    '''
    return Bodies(
        x=bodies.x[mask],
        dx=bodies.dx[mask],
        b=None if bodies.b is None else bodies.b[mask])


def _update_bodies(bodies, a, db, dt, cull_on_exit):
    '''Compute the movement update for 'bodies'.

    bodies -- astro.Bodies -- to update

    a -- array(N x 2) -- acceleration

    db -- array(N) -- change in bearing

    dt -- float -- timestep

    cull_on_exit -- bool -- if True, remove out-of-bounds bodies rather than
                    wrapping

    returns -- astro.Bodies
    '''
    # the approximation (dx + dx') / 2 for updating position seems to lead
    # to instability, so just using dx' here
    dx = bodies.dx + a * dt
    x = bodies.x + dt * dx
    b = None if bodies.b is None else bodies.b + db
    if cull_on_exit:
        return _mask(
            Bodies(x=x, dx=dx, b=b),
            mask=(([-1, -1] <= x) & (x <= [1, 1])).any(axis=1))
    else:
        return Bodies(x=_wrap_unit_square(x), dx=dx, b=b)


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

    # fire bullets
    next_reload = state.reload + config.dt
    next_bullets = _mask(
        state.bullets,
        ~collisions[(nships + nplanets):])
    if config.reload_time <= next_reload:
        ships = state.ships
        ships_direction = _direction(ships.b)
        next_bullets = Bodies(
            x=np.concatenate([
                next_bullets.x,
                ships.x + 1.001 * config.ship_radius * ships_direction
            ], axis=0),
            dx=np.concatenate([
                next_bullets.dx,
                ships.dx + config.bullet_speed * ships_direction
            ], axis=0),
            b=None)
        next_reload -= config.reload_time

    next_state = State(
        ships=_update_bodies(
            state.ships,
            a=ships_a,
            db=ships_db,
            dt=config.dt,
            cull_on_exit=False),
        planets=_update_bodies(
            state.planets,
            a=_gravity(state.planets, state.planets.x, config=config),
            db=None,
            dt=config.dt,
            cull_on_exit=False),
        bullets=_update_bodies(
            next_bullets,
            a=0,
            db=None,
            dt=config.dt,
            cull_on_exit=True),
        reload=next_reload,
        t=state.t + config.dt)
    return next_state, np.zeros(nships, dtype=np.float32)


def swap_ships(state):
    '''Swap the two ships in "state", so that bots can play against each other.
    '''
    s = state.ships
    return State(
        ships=Bodies(x=s.x[::-1], dx=s.dx[::-1], b=s.b[::-1]),
        planets=state.planets,
        bullets=state.bullets,
        reload=state.reload,
        t=state.t)


def play(bot_0, bot_1, config):
    '''Play out the game between bot_0 & bot_1.

    bot_0, bot_1 -- bots to play

    config -- astro.Config

    returns -- int or None -- 0 if bot_0 won, 1 if bot_1 won, None if a draw
    '''
    state = create(config)
    while True:
        control = np.array([bot_0(state), bot_1(swap_ships(state))])
        state, reward = step(state, control, config)
        if state is None:
            if reward[1] < reward[0]:
                return 0
            elif reward[0] < reward[1]:
                return 1
            else:
                return None


def find_better(games, max_trials, threshold):
    '''Find the better bot from a series of game results.

    games -- iterable(int or None) -- results of play()

    max_trials -- int -- trial limit before giving up

    threshold -- float -- how sure must we be of the answer?

    returns -- int -- 0 if the first player is better, 1 if the second player,
                      None if inconclusive
    '''
    nwins0 = 0
    nwins1 = 0
    for n, winner in enumerate(games):
        nwins0 += (winner == 0)
        nwins1 += (winner == 1)
        p0 = _pbetter(nwins=nwins0, nlosses=nwins1)
        if threshold < p0:
            # confident that 0 is better
            return 0
        elif threshold < (1 - p0):
            # confident that 1 is better
            return 1
        else:
            nremaining = max_trials - n - 1
            if ((_pbetter(nwins=nwins0 + nremaining, nlosses=nwins1) <
                 threshold) and
                ((1 - _pbetter(nwins=nwins0, nlosses=nwins1 + nremaining)) <
                 threshold)):
                # inconclusive - not enough trials left to reach confidence
                return None


class ScriptBot:
    '''Tries to stay alive first, otherwise tries to shoot you.
    '''
    # these values optimized using a random walk 1-vs-1 tournament
    DEFAULT_ARGS = dict(
        avoid_distance=0.1,
        avoid_threshold=0.45,
    )

    def __init__(self, args, config):
        self.args = args
        self.config = config

    def _fly_to(self, state, b, t, fwd):
        angle = _norm_angle(b - state.ships.b[0])
        if angle < -t:
            return 0  # rotate left
        elif t < angle:
            return 4  # rotate right
        elif fwd:
            return 3
        else:
            return 2

    def _danger(self, x, dx, b):
        '''Am I in danger of crashing into this planet?

        x -- array(2) -- my position, relative to planet

        dx -- array(2) -- my velocity, relative to planet

        b -- float -- my bearing

        returns -- float or None -- bearing to take action, or `None`
                   if no action is required
        '''
        radius = (self.config.planet_radius + self.config.ship_radius)
        b = 2 * _dot(_norm(dx), x)
        c = (_mag(x) ** 2 - (radius + self.args['avoid_distance']) ** 2)
        det = b ** 2 - 4 * c
        if 0 < det and 0 <= -b + np.sqrt(det):
            # real-valued roots & at least one positive root
            distance = -b - np.sqrt(det)
            rotation = abs(_norm_angle(_bearing(x) - b))
            speed = _mag(dx)
            if distance < (speed / self.config.ship_thrust +
                           self.config.ship_rspeed / rotation) * speed:
                return _bearing(x)
        return None

    def __call__(self, state):
        # don't crash into planets
        for i in range(state.planets.x.shape[0]):
            b = self._danger(state.ships.x[0] - state.planets.x[i],
                             state.ships.dx[0] - state.planets.dx[i],
                             state.ships.b[0])
            if b is not None:
                return self._fly_to(state, b, self.args['avoid_threshold'],
                                    fwd=True)

        # aim for the enemy
        enemy_distance = _mag(state.ships.x[1] - state.ships.x[0])
        bullet_time = (enemy_distance / self.config.bullet_speed)
        enemy_forecast = (state.ships.x[1] +
                          bullet_time *
                          (state.ships.dx[1] - state.ships.dx[0]))
        return self._fly_to(
            state,
            _bearing(enemy_forecast - state.ships.x[0]),
            self.config.ship_radius / enemy_distance,
            fwd=False)

    @staticmethod
    def mutate(args, scale, random):
        '''Generate a local random mutation of `args`.

        args -- dict -- arguments

        scale -- float -- base scale of mutation

        random -- numpy.random.RandomState -- random generator

        returns -- dict -- mutated arguments
        '''
        args = args.copy()
        args['avoid_distance'] += scale * random.normal()
        args['avoid_threshold'] += scale * random.normal()
        return args

    @classmethod
    def play_many(cls, baseline_args, args, config, random):
        '''Return an iterable of games played between two ScriptBots.

        baseline_args -- dict -- args to use as a baseline

        args -- dict -- args to compete

        config -- astro.Config -- basic config to use

        random -- numpy.random.RandomState -- generator

        returns -- iterable(int or None) -- game winners
        '''
        while True:
            config = config._replace(seed=random.randint(1 << 30))
            yield play(cls(baseline_args, config), cls(args, config), config)

    @classmethod
    def optimize(cls, config, max_trials, threshold, max_mutations, scale):
        '''Optimize the arguments of this class, returning the best arguments.
        '''
        random = np.random.RandomState(config.seed)
        best = cls.DEFAULT_ARGS
        sys.stderr.write('Initial {}\n'.format(best))
        for _ in range(max_mutations):
            candidate = cls.mutate(best, scale=scale, random=random)
            winner = find_better(
                cls.play_many(baseline_args=best, args=candidate,
                              config=config, random=random),
                max_trials=max_trials,
                threshold=threshold)
            sys.stderr.write('Trial {} -> {}\n'.format(candidate, winner))
            if winner == 1:
                best = candidate
        return best


class NothingBot:
    '''Doesn't do anything!
    '''
    def __call__(self, state):
        return 2


class QBot(C.Chain):
    '''Deep Q learning RL bot.
    '''
    @staticmethod
    def get_features(state):
        '''Create the feature vector from the current state.
        A little feature engineering here to make the model's job easier
        - concatenate the ships feature vector onto every other object.

        state -- astro.State

        returns -- array(N x D) -- feature array (floats)
        '''
        def feature(bodies):
            return np.concatenate(
                (bodies.x, bodies.dx) +
                (()
                 if bodies.b is None else
                 (_norm_angle(bodies.b[:, np.newaxis]),)),
                axis=1)

        nships = state.ships.x.shape[0]
        nplanets = state.planets.x.shape[0]
        nbullets = state.bullets.x.shape[0]

        # dim 0: [planets, bullets]
        # dim 1: [type, ships, object]
        features = np.zeros(
            (nplanets + nbullets, 1 + (5 * nships) + 4),
            dtype=np.float32)
        features[:nplanets, 0] = 1  # planet flag
        features[nplanets:, 0] = -1  # bullet flag
        robject = 1 + 5 * nships
        features[:, 1:robject] = feature(state.ships).flatten()
        features[:nplanets, robject:] = feature(state.planets)
        features[nplanets:, robject:] = feature(state.bullets)

        return features

    def __init__(self):
        super().__init__()
        with self.init_scope():
            nf, nq = 15, 6
            nf0, nf1, nq0 = 32, 32, 32
            self.activation = C.functions.elu
            self.f0 = C.links.Linear(nf, nf0)
            self.f1 = C.links.Linear(nf0, nf1)
            self.pool = C.functions.max
            self.q0 = C.links.Linear(nf1, nq0)
            self.q1 = C.links.Linear(nq0, nq)

    def __call__(self, state):
        features = C.Variable(self.get_features(state))
        f1 = self.f1(self.activation(self.f0(features)))
        pool = self.pool(f1, axis=0).reshape(1, -1)
        q = C.functions.tanh(
            self.q1(self.activation(self.q0(pool)))
        ).reshape(-1)
        print(q.data)
        return None  # TODO


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
