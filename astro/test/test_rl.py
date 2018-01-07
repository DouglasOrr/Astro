from .. import rl, util, core
import numpy as np
import torch as T
import itertools as it


def change_to_crash(config, state, random):
    '''Change a single "solo easy" state to a "crash" state.'''
    assert state.ships.x.shape[0] == 1
    x = config.planet_radius * util.direction(random.uniform(0, 2 * np.pi))
    # keep ships dx=0 (for now)
    # bearing has already been randomly initialized
    return state._replace(ships=state.ships._replace(x=x.reshape(1, -1)))


def change_to_noncrash(config, state, random):
    '''Change a single "solo easy" state to a random "not crashed" state.'''
    assert state.ships.x.shape[0] == 1
    assert state.planets.x.shape[0] == 1
    # use rejection sampling to generate a "non-crash"
    x = next(s
             for _ in it.repeat(None)
             for s in [random.uniform(-1, 1, 2).astype(np.float32)]
             if config.planet_radius < util.mag(s - state.planets.x[0]))
    # keep ships dx=0 (for now)
    # bearing has already been randomly initialized
    return state._replace(ships=state.ships._replace(x=x.reshape(1, -1)))


def gen_crash_states(config, ncrash, n, random):
    '''Generate fake states & rewards for an imaginary set of crashes &
    non-crashes, for testing learnability.

    config -- astro.Config -- base config to use

    ncrash -- int -- how many crashed states to generate

    n -- int -- how many overall states to generate

    random -- np.random.RandomState

    returns -- [astro.State], array(n) -- states, rewards
    '''
    if not config.solo and config.max_planets == 1:
        raise ValueError(
            'gen_crash_states only supports solo configurations'
            ' with one planet, for now')

    states = []
    for i in range(n):
        state = core.create(config._replace(seed=random.randint(1 << 30)))
        if i < ncrash:
            states.append(change_to_crash(config, state, random))
        else:
            states.append(change_to_noncrash(config, state, random))
    return (states,
            np.concatenate((
                np.full(ncrash, -1, dtype=np.float32),
                np.full(n - ncrash, 0, dtype=np.float32))))


def test_convergence():
    weighted = True
    nreplay = 20
    ncrashes, nsamples = 1, 100
    nsteps, vinterval = 100, 10
    nvcrashes, nvsamples = 1000, 2000

    random = np.random.RandomState(0)
    T.manual_seed(0)
    network = rl.ValueNetwork(solo=True, nout=1)
    valid, valid_y = gen_crash_states(
        core.SOLO_EASY_CONFIG, nvcrashes, nvsamples, random)
    valid_y = T.autograd.Variable(T.FloatTensor(valid_y))
    replay_buffer = []
    for step in range(nsteps):
        replay_buffer.append(gen_crash_states(
            core.SOLO_EASY_CONFIG, ncrashes, nsamples, random))

        x = [x for xs, y in replay_buffer[-nreplay:] for x in xs]
        y = np.concatenate([y for x, y in replay_buffer[-nreplay:]])

        losses = ((network.evaluate_batch(x).view(-1) -
                   T.autograd.Variable(T.FloatTensor(y))) ** 2)
        if weighted:
            crashes = (y < 0)
            weights = T.autograd.Variable(T.FloatTensor(
                np.where(crashes,
                         0.5 / crashes.sum(),
                         0.5 / (y.size - crashes.sum()))))
        else:
            weights = 1. / y.size
        network.opt.zero_grad()
        (losses * weights).sum(0).backward()
        network.opt.step()

        if step % vinterval == 0:
            vlosses = ((network.evaluate_batch(valid).view(-1) - valid_y) ** 2)
            print('{} {} -- {:.3f} -- {:.3f} {:.3f} ({:.0%}, {:.0%})'.format(
                step, (step + 1) * nsamples,
                float(vlosses.mean()),
                float(vlosses[:nvcrashes].mean()),
                float(vlosses[nvcrashes:].mean()),
                float((vlosses[:nvcrashes] < 0.25).type(T.FloatTensor).mean()),
                float((vlosses[nvcrashes:] < 0.25).type(T.FloatTensor).mean()),
            ))

    # Check 90% classification accuracy for crashes & 80% for noncrashes
    vlosses = (
        (network.evaluate_batch(valid).view(-1) - valid_y) ** 2
    ).data.numpy()
    assert 0.9 < (vlosses[:nvcrashes] < 0.25).mean()
    assert 0.8 < (vlosses[nvcrashes:] < 0.25).mean()
