import importlib
import json
import numpy as np
import scipy.stats
import cProfile
import contextlib


# to/from JSON

def to_jsonable(obj):
    '''Convert `obj` to a JSON-able object, adding support for namedtuple &
    numpy.array.
    '''
    if hasattr(obj, '_asdict'):
        d = to_jsonable(obj._asdict())
        _type = type(obj)
        d['_type'] = '{}:{}'.format(_type.__module__, _type.__name__)
        return d
    elif isinstance(obj, np.ndarray):
        return {'_values': obj.tolist(),
                '_shape': obj.shape}
    elif isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (tuple, list)):
        return list(to_jsonable(x) for x in obj)
    else:
        return obj


def from_jsonable(obj):
    '''Convert `obj` from a JSON-able object, adding support for namedtuple &
    numpy.array.
    '''
    if isinstance(obj, (list, tuple)):
        return list(from_jsonable(x) for x in obj)
    elif isinstance(obj, dict):
        if obj.keys() == {'_values', '_shape'}:
            a = np.array(obj['_values']).reshape(obj['_shape'])
            return a.astype(np.float32) if a.dtype is np.float64 else a
        elif '_type' in obj.keys():
            _module, _name = obj.pop('_type').split(':')
            return getattr(
                importlib.import_module(_module), _name
            )(**from_jsonable(obj))
        else:
            return {k: from_jsonable(v) for k, v in obj.items()}
    else:
        return obj


def to_json(obj):
    '''Return a JSON string for `obj`, with support for namedtuple & numpy.array.
    '''
    return json.dumps(to_jsonable(obj))


def from_json(s):
    '''Load an object from a JSON string, with support for namedtuple &
    numpy.array.
    '''
    return from_jsonable(json.loads(s))


# Maths/geometry

def direction(bearing):
    '''Return a unit vector pointing in the direction 'bearing' from +y.
    '''
    return np.stack((np.sin(bearing, dtype=np.float32),
                     np.cos(bearing, dtype=np.float32)),
                    axis=-1)


def bearing(x):
    '''Return the bearing of the vector x.

    x -- array(... x 2) -- direction vectors

    returns -- array(...) -- bearings from +y
    '''
    return np.arctan2(x[..., 0], x[..., 1])


def mag(x):
    '''Compute the magnitude of x along the last dimension.

    x -- array(... x 2)

    returns -- array(...)
    '''
    return np.sqrt((x ** 2).sum(axis=-1))


def norm(x):
    '''Normalize x along the last dimension.

    x -- array(... x 2)

    returns -- array(... x 2)
    '''
    return x / (mag(x) + 1e-12)[..., np.newaxis]


def norm_angle(b):
    '''Normalize an angle (/bearing) to the range [-pi, +pi].

    b -- float -- bearing

    returns -- float
    '''
    return ((b + np.pi) % (2 * np.pi)) - np.pi


def dot(a, b):
    '''Return the dot product between batches of vectors.

    a, b -- array(... x N)

    returns -- array(...)
    '''
    return (a * b).sum(axis=-1)


def wrap_unit_square(x):
    '''Wraps x around the unit square.
    '''
    return ((x + 1) % 2) - 1


# Other

def p_better(nwins, nlosses):
    '''Compute the probability of this being better, given the number
    of wins & losses, under an informative beta prior.

    nwins -- int -- number of wins

    nlosses -- int -- number of losses

    returns -- float -- probability of the underlying win rate being
               greater than 0.5
    '''
    return 1 - scipy.stats.beta.cdf(0.5, 1 + nwins, 1 + nlosses)


def find_better(games, max_trials, threshold):
    '''Find the better bot from a (lazy) sequence of game results.

    games -- iterable(int or None) -- iterable of first result of play()

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
        p0 = p_better(nwins=nwins0, nlosses=nwins1)
        if threshold < p0:
            # confident that 0 is better
            return 0
        elif threshold < (1 - p0):
            # confident that 1 is better
            return 1
        else:
            nremaining = max_trials - n - 1
            if ((p_better(nwins0 + nremaining, nwins1) <
                 threshold) and
                ((1 - p_better(nwins0, nwins1 + nremaining)) <
                 threshold)):
                # inconclusive - not enough trials left to reach confidence
                return None


@contextlib.contextmanager
def profiling(filename):
    '''Wrap some code in with_profile to start/stop cProfile, then dump the
    results to the file.
    '''
    p = cProfile.Profile()
    p.enable()
    yield
    p.disable()
    p.dump_stats(filename)
