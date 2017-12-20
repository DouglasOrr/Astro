'''Astro - simple tiny physics simulation for trying out reinforcement
learning.
'''

import collections
import numpy as np


Body = collections.namedtuple(
    'Body',
    ('x', 'dx', 'angle', 'mass', 'radius', 'ttl'))

State = collections.namedtuple(
    'State',
    ('me', 'enemy', 'bodies'))


def create(n, seed):
    '''Create a new game state randomly.

    Currently only supports the "binary stars" map - two equal mass & size
    stars in orbit of one another.

    n -- int -- how many copies to create

    seed -- int -- random seed
    '''
    random = np.random.RandomState(seed)

    # config
    gravity = 0.5
    planet_orbit = 0.5
    planet_mass = 1
    planet_radius = 0.1
    ship_position = np.array([0.9, 0.9], dtype=np.float32)
    ship_radius = 0.01

    # ships
    ship_a = (ship_position[np.newaxis, :] *
              np.sign(random.rand(n, 2).astype(np.float32) - 0.5))
    ship_b = -ship_a

    # planets
    orientation = np.pi * random.rand(n).astype(np.float32)
    planet_a = planet_orbit * np.stack(
        (np.cos(orientation), np.sin(orientation)),
        axis=1)
    planet_a_dx = np.sqrt(gravity * planet_mass / 2) * np.stack(
        (-np.sin(orientation), np.cos(orientation)),
        axis=1)
    planet_b = -planet_a
    planet_b_dx = -planet_a_dx

    return State(
        Body(ship_a,
             np.zeros((n, 2), dtype=np.float32),
             np.zeros(n, dtype=np.float32),
             np.zeros(n, dtype=np.float32),
             np.full(n, ship_radius, dtype=np.float32),
             np.full(n, np.inf, dtype=np.float32)),
        Body(ship_b,
             np.zeros((n, 2), dtype=np.float32),
             np.zeros(n, dtype=np.float32),
             np.zeros(n, dtype=np.float32),
             np.full(n, ship_radius, dtype=np.float32),
             np.full(n, np.inf, dtype=np.float32)),
        Body(np.stack((planet_a, planet_b), axis=1),
             np.stack((planet_a_dx, planet_b_dx), axis=1),
             np.zeros((n, 2), dtype=np.float32),
             planet_mass * np.ones((n, 2), dtype=np.float32),
             planet_radius * np.ones((n, 2), dtype=np.float32),
             np.full((n, 2), np.inf, dtype=np.float32)),
    )


def step(state, dt, gravity):
    pass


def test_nothing():
    pass
