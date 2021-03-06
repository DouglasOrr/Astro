"""Astro - simple tiny physics simulation for trying out reinforcement
learning.
"""

import numpy as np
import collections
import os
from . import util


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
        'solo',
        'outer_ship_position',
        'inner_ship_position',
        'max_planets',
        'planet_orbit',
        'planet_mass',
        'planet_radius',
    ))

Tick = collections.namedtuple(
    'Tick',
    ('state', 'control', 'reward', 'bot_data'))

Game = collections.namedtuple(
    'Game',
    ('config', 'winner', 'ticks'))


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
    solo=False,
    inner_ship_position=0.2,
    outer_ship_position=0.9,
    max_planets=4,
    planet_orbit=0.5,
    planet_mass=1.0,
    planet_radius=0.2,
)
SOLO_CONFIG = DEFAULT_CONFIG._replace(solo=True, reload_time=1000)
SOLO_EASY_CONFIG = SOLO_CONFIG._replace(max_planets=1)


def generate_configs(config):
    """Generate an infinite sequence of differently seeded configurations from
    a single seed configuration.
    """
    random = np.random.RandomState(config.seed)
    while True:
        yield config._replace(seed=random.randint(1 << 30))


def create(config):
    """Create a new game state randomly.
    """
    random = np.random.RandomState(config.seed)
    nplanets = random.randint(1, config.max_planets + 1)

    # 1. ships
    outer = (config.outer_ship_position *
             np.sign(random.rand(2).astype(np.float32) - 0.5))
    inner = (config.inner_ship_position *
             util.direction(2 * np.pi * random.rand()))
    if nplanets == 1 and config.solo:
        ships = outer[np.newaxis, ...]
    elif nplanets == 1:
        ships = np.stack((outer, -outer), axis=0)
    elif config.solo:
        ships = (outer if random.rand() < 0.5 else inner)[np.newaxis, ...]
    else:
        # randomly choose middle/outer initialization for each ship
        ships = np.stack(
            (outer, inner) if random.rand() < 0.5 else (inner, outer),
            axis=0)
    ships_dx = np.zeros_like(ships)
    ships_b = 2 * np.pi * random.rand(*ships.shape[:-1]).astype(np.float32)

    # 2. planets
    if nplanets == 1:
        planets = np.zeros((1, 2), dtype=np.float32)
        planets_dx = np.zeros_like(planets)
    else:
        orientation = (2 * np.pi * random.rand() +
                       np.linspace(0, 2 * np.pi, num=nplanets, endpoint=False))
        reverse = random.choice((-1, 1))
        planets = config.planet_orbit * util.direction(orientation)
        planets_dx = (
            np.sqrt(config.gravity * config.planet_mass * (nplanets - 1) / 2) *
            util.direction(orientation + reverse * np.pi / 2))

    # 3. aggregate everything together into state
    return State(
        ships=Bodies(x=ships, dx=ships_dx, b=ships_b),
        planets=Bodies(x=planets, dx=planets_dx, b=None),
        bullets=Bodies(
            x=np.zeros((0, 2), dtype=np.float32),
            dx=np.zeros((0, 2), dtype=np.float32),
            b=None
        ),
        reload=0.0,
        t=0.0,
    )


def _gravity(planets, x, config):
    """Compute the acceleration due to gravity of 'planets' on objects at position
    'x'.

    planets -- astro.Bodies -- only massive bodies

    x -- array(N x 2) -- positions to evaluate gravitational field

    config -- astro.Config -- constants

    returns -- array(N x 2) -- gravitational field (force per unit mass)
    """
    rx = planets.x[np.newaxis, :, :] - x[:, np.newaxis, :]
    f_rx = (config.gravity * config.planet_mass /
            np.maximum(1e-12, (rx ** 2).sum(axis=2)))
    return (f_rx[:, :, np.newaxis] * rx).sum(axis=1)


def _mask(bodies, mask):
    """Only select bodies for which mask is true.

    bodies -- astro.Bodies

    mask -- np.array(N; bool) -- selects bodies to keep

    returns -- astro.Bodies
    """
    return Bodies(
        x=bodies.x[mask],
        dx=bodies.dx[mask],
        b=None if bodies.b is None else bodies.b[mask])


def _update_bodies(bodies, a, db, dt, cull_on_exit):
    """Compute the movement update for 'bodies'.

    bodies -- astro.Bodies -- to update

    a -- array(N x 2) -- acceleration

    db -- array(N) -- change in bearing

    dt -- float -- timestep

    cull_on_exit -- bool -- if True, remove out-of-bounds bodies rather than
                    wrapping

    returns -- astro.Bodies
    """
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
        return Bodies(x=util.wrap_unit_square(x), dx=dx, b=b)


def _collisions(x, r):
    """Compute a collision mask between the objects at positions 'xs', with
    radiuses 'rs'.

    x -- array(N x 2) -- positions

    r -- array(N) -- radiuses

    returns -- array(N; bool) -- collisions
    """
    rx2 = ((x[np.newaxis, :, :] - x[:, np.newaxis, :]) ** 2).sum(axis=2)
    r2 = (r[np.newaxis, :] + r[:, np.newaxis]) ** 2
    return ((rx2 < r2) & ~np.eye(x.shape[0], dtype=np.bool)).any(axis=1)


def step(state, control, config):
    """Advance the world state by a single tick of the game clock.

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
    """
    ships_a = (config.ship_thrust *
               (control % 2)[:, np.newaxis] *
               util.direction(state.ships.b) +
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
        # (in solo games, this counts as a win)
        return None, np.full(nships, 1 if config.solo else 0, dtype=np.float32)

    # fire bullets
    next_reload = state.reload + config.dt
    next_bullets = _mask(
        state.bullets,
        ~collisions[(nships + nplanets):])
    if config.reload_time <= next_reload:
        ships = state.ships
        ships_direction = util.direction(ships.b)
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


def roll_ships(state, index):
    """Roll the ships in "state", so that bots can play against each other,
    and "my bot" is always at index 0.

    state -- astro.State or None

    index -- int -- index that should be moved to zero

    returns -- astro.State or None
    """
    if state is None:
        return None
    s = state.ships
    return State(
        ships=Bodies(
            x=np.roll(s.x, -index, 0),
            dx=np.roll(s.dx, -index, 0),
            b=np.roll(s.b, -index, 0)),
        planets=state.planets,
        bullets=state.bullets,
        reload=state.reload,
        t=state.t)


class Bot:
    def __call__(self, state):
        """Called to get the bot's control output for a given state.

        state -- astro.State

        returns -- int -- control signal (see `step`)
        """
        raise NotImplementedError

    def reward(self, state, reward):
        """Called to inform the bot of a reward in the given new state.

        state -- astro.State

        reward -- float -- game reward signal
        """
        pass

    @property
    def data(self):
        """Get internal data from the bot, to be saved alongside the log,
        if available.

        returns -- jsonable object
        """
        pass


class Bots:
    @staticmethod
    def control(bots, state):
        return np.array([bot(roll_ships(state, index))
                         for index, bot in enumerate(bots)])

    @staticmethod
    def reward(bots, state, reward):
        for index, bot in enumerate(bots):
            if hasattr(bot, 'reward'):
                bot.reward(roll_ships(state, index), reward[index])

    @staticmethod
    def data(bots):
        return [bot.data if hasattr(bot, 'data') else None
                for bot in bots]


def play(config, bots):
    """Play out the game between bot_0 & bot_1.

    config -- astro.Config

    bots -- [Bot] -- bots to play, where each bot should be like `astro.Bot`

    returns -- astro.Game -- including game outcome
    """
    ticks = []
    state = create(config)
    while True:
        # 1. Gather input & record state
        control = Bots.control(bots, state)

        # 2. Advance the game physics
        old_state = state
        state, reward = step(state, control, config)

        # 3. Provide rewards
        Bots.reward(bots, state, reward)

        # 4. Save game ticks & handle end-of-game
        ticks.append(Tick(
            state=old_state,
            control=control,
            reward=reward,
            bot_data=Bots.data(bots)))
        if state is None:
            return Game(
                config=config,
                ticks=ticks,
                winner=None if np.max(reward) < 1 else int(np.argmax(reward)),
            )


def save_log(path, game):
    """Saves a game log as jsonlines.

    path -- string -- file path

    game -- astro.Game
    """
    dir_ = os.path.dirname(path)
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    with open(path, 'w') as f:
        f.write(util.to_json(
            dict(config=game.config, winner=game.winner)) + '\n')
        for tick in game.ticks:
            f.write(util.to_json(tick) + '\n')


def load_log(path):
    """Load a game log from file.

    path -- string -- file path

    returns -- astro.Game
    """
    with open(path, 'r') as f:
        header = util.from_json(next(f))
        ticks = [util.from_json(line) for line in f]
        return Game(
            config=header['config'],
            winner=header['winner'],
            ticks=ticks)
