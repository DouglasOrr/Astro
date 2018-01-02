import json
import collections
import numpy as np
import itertools as it
from .. import util


TypeForTest = collections.namedtuple(
    'TypeForTest',
    ('foo', 'bar'))


def test_to_from_json():
    for obj in [
            123,
            'abc',
            dict(a=1, b=False, c=None),
            TypeForTest('def', 456),
            [dict(thing=TypeForTest('ghi', 789))],
    ]:
        s = util.to_json(obj)
        json.loads(s)
        assert util.from_json(s) == obj


def test_direction():
    np.testing.assert_allclose(util.direction(0), [0, 1], atol=1e-7)
    np.testing.assert_allclose(
        util.direction(np.arange(0, 2 * np.pi, np.pi/2)),
        [[0, 1],
         [1, 0],
         [0, -1],
         [-1, 0]], atol=1e-7)


def test_bearing():
    np.testing.assert_allclose(util.bearing(np.array([0, 1])), 0, atol=1e-7)
    np.testing.assert_allclose(
        util.bearing(np.array([
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0]])),
        [0, np.pi / 2, np.pi, -np.pi / 2])


def test_mag_norm_angle():
    np.testing.assert_allclose(util.mag(np.array([3, 4])), 5)
    np.testing.assert_allclose(util.norm(np.array([3, 4])), [0.6, 0.8])
    np.testing.assert_allclose(util.norm_angle(2 * np.pi + 0.5), 0.5)
    np.testing.assert_allclose(util.norm_angle(-4 * np.pi - 0.5), -0.5)


def test_wrap_unit_square():
    np.testing.assert_allclose(
        util.wrap_unit_square(np.array([
            [1.01, -0.95],
            [0.95, -1.01],
        ])),
        np.array([
            [-0.99, -0.95],
            [0.95, 0.99]
        ]))


def test_find_better():
    assert util.find_better(it.repeat(0), max_trials=100, threshold=0.99) == 0
    assert util.find_better(it.repeat(1), max_trials=100, threshold=0.99) == 1
    assert util.find_better(
        it.cycle([0, 1]), max_trials=1000, threshold=0.99) is None
