import json
import collections
import numpy as np
import itertools as it
import os
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


def _check_batchable(fn, inputs, outputs, atol=1e-6):
    xs, ys = np.array(inputs), np.array(outputs)
    for x, y in zip(xs, ys):
        # single example
        np.testing.assert_allclose(fn(x), y, atol=atol)

    # 1D batch
    batch_x, batch_y = np.stack(xs), np.stack(ys)
    np.testing.assert_allclose(fn(batch_x), batch_y, atol=atol)

    # 2D batch (1xN & Nx1)
    np.testing.assert_allclose(
        fn(batch_x[np.newaxis, ...]),
        batch_y[np.newaxis, ...],
        atol=atol)
    np.testing.assert_allclose(
        fn(batch_x[:, np.newaxis, ...]),
        batch_y[:, np.newaxis, ...],
        atol=atol)


def test_geometry():
    _check_batchable(util.direction,
                     np.arange(0, 2 * np.pi + 1e-3, np.pi/2),
                     [[0, 1],
                      [1, 0],
                      [0, -1],
                      [-1, 0],
                      [0, 1]])

    _check_batchable(util.bearing,
                     [[0, 1],
                      [1, 0],
                      [0, -1],
                      [-1, 0]],
                     [0, np.pi / 2, np.pi, -np.pi / 2])

    _check_batchable(util.mag,
                     [[0, 1], [-100, 0], [3, 4]],
                     [1, 100, 5])

    _check_batchable(util.norm,
                     [[0, 1], [-100, 0], [3, 4]],
                     [[0, 1], [-1, 0], [0.6, 0.8]])

    _check_batchable(util.norm_angle,
                     [2 * np.pi + 0.5, -4 * np.pi - 0.5],
                     [0.5, -0.5])

    _check_batchable(lambda x: util.dot(x, x),
                     [[0, -1], [2, 3]],
                     [1, 13])

    _check_batchable(util.wrap_unit_square,
                     [[1.01, -0.95], [0.95, -1.01]],
                     [[-0.99, -0.95], [0.95, 0.99]])


def test_better():
    np.testing.assert_allclose(util.p_better(0, 0), 0.5)
    np.testing.assert_allclose(util.p_better(5, 5), 0.5)
    assert 0.5 < util.p_better(nwins=6, nlosses=5)
    assert util.p_better(nwins=5, nlosses=6) < 0.5

    assert util.find_better(it.repeat(0), max_trials=100, threshold=0.99) == 0
    assert util.find_better(it.repeat(1), max_trials=100, threshold=0.99) == 1
    assert util.find_better(
        it.cycle([0, 1]), max_trials=1000, threshold=0.99) is None


def test_profiling():
    with util.profiling('/tmp/test_profiling.cprof'):
        sum(range(10000))
    assert os.path.isfile('/tmp/test_profiling.cprof')
