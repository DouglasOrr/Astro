import numpy as np
import torch as T
import itertools as it
import functools as ft
import sys
from . import util, core


class EpsilonGreedy:
    '''A stateful bot which returns an action according to a random policy, or
    `None` if the random policy should not be active.
    '''
    def __init__(self, t_in, t_out, seed):
        self.t_in, self.t_out = t_in, t_out
        self._random = np.random.RandomState(seed)
        self._t = 0
        self._policy = None

    def __call__(self, state):
        dt = state.t - self._t
        if (self._policy is None and
                np.exp(-dt / self.t_in) < self._random.rand()):
            self._policy = self._random.randint(0, 5)
        elif (self._policy is not None and
              np.exp(-dt / self.t_out) < self._random.rand()):
            self._policy = None
        self._t = state.t
        return self._policy


class ValueNetwork(T.nn.Module):
    '''Deep Q learning RL network for playing astro.
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
                 (util.norm_angle(bodies.b[:, np.newaxis]) / np.pi,)),
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

    def __init__(self, solo, nout):
        super().__init__()
        nf = (10 if solo else 15)
        nh, d = 32, 1
        self.activation = T.nn.functional.softsign
        self.f0 = T.nn.Linear(nf, nh)
        self.f = T.nn.ModuleList([T.nn.Linear(nh, nh) for _ in range(d)])
        self.pool = lambda x: T.max(x, -2)[0]
        self.v = T.nn.ModuleList([T.nn.Linear(nh, nh) for _ in range(d)])
        self.v0 = T.nn.Linear(nh, nout)

        self.opt = T.optim.Adam(self.parameters(), lr=1e-2)
        # self.opt = T.optim.Rprop(self.parameters())
        # self.opt = T.optim.RMSprop(self.parameters())
        # self.opt = T.optim.Adagrad(self.parameters())
        # self.opt = T.optim.SGD(self.parameters(), lr=1e-1)

    def forward(self, x):
        f = ft.reduce(
            lambda a, layer: layer(self.activation(a)),
            self.f, self.f0(x))
        v = ft.reduce(
            lambda a, layer: self.activation(layer(a)),
            self.v, self.pool(f))
        return T.tanh(self.v0(v))

    def evaluate(self, state):
        return self(T.autograd.Variable(
            T.FloatTensor(
                self.get_features(state))))


class QBot(core.Bot):
    '''A basic "evaluation mode" Q-learning bot with no epsilon-greedy
    policy & no training.
    '''
    def __init__(self, network):
        self.q = network
        self._last_q = None

    def __call__(self, state):
        self._last_q = self.q.evaluate(state).data.numpy()
        return np.argmax(self._last_q)

    @property
    def data(self):
        return dict(q=self._last_q)


class QBotTrainer(QBot):
    def __init__(self, network, seed):
        super().__init__(network)
        self.greedy = EpsilonGreedy(t_in=1.0, t_out=0.1, seed=seed)
        self.n_steps = 100  # a bit high!
        self.discount = 0.98
        self._qbuffer = []
        self._data = dict(greedy=False, q=None)

    def __call__(self, state):
        # Act according to an e-greedy policy or Q
        q = self.q.evaluate(state)
        greedy = self.greedy(state)
        action = (greedy if greedy is not None else np.argmax(q.data.numpy()))
        self._qbuffer.append(q[action])
        self._data['greedy'] = greedy is not None
        self._data['q'] = q.data.numpy()
        return action

    def reward(self, new_state, reward):
        if new_state is None or self.n_steps <= len(self._qbuffer):
            # we know there are only rewards for terminating states, in this
            # game, so no need to store any others
            if new_state is None:
                r = reward
            else:
                r = self.discount * np.max(
                    self.q.evaluate(new_state).data.numpy())
            target = T.autograd.Variable(T.FloatTensor(
                r * (self.discount ** np.arange(len(self._qbuffer)))))

            q = T.cat(self._qbuffer[::-1])
            # w = 0.1 + 0.9 * T.abs(target)  # TODO: crude weighting
            e = ((target - q) ** 2).sum(0)
            self.q.opt.zero_grad()
            (e / self.n_steps).backward()
            self.q.opt.step()
            self._data['step'] = dict(
                loss=float(e.data), n=len(self._qbuffer))
            self._qbuffer.clear()
        else:
            if 'step' in self._data:
                del self._data['step']

    @property
    def data(self):
        return self._data.copy()

    @staticmethod
    def average_step_loss(steps):
        '''Compute the average (weighted) loss from a sequence of steps.
        '''
        loss, n = 0, 0
        for step in steps:
            loss += step['loss']
            n += step['n']
        return loss / n


def train(config, interval, limit, log_prefix=None):
    network = ValueNetwork(solo=config.solo, nout=6)
    train_bots = [QBotTrainer(network, seed=n)
                  for n in range(1 if config.solo else 2)]
    eval_bots = [QBot(network)
                 for _ in range(1 if config.solo else 2)]

    games = []
    for n, config in it.islice(enumerate(core.generate_configs(config)),
                               limit):
        if n % interval == 0:
            # Validate
            valid_game = core.play(config, eval_bots)
            if log_prefix is not None:
                core.save_log(log_prefix + '.{}.log'.format(n), valid_game)

            msg = 'N: {}'.format(n)
            msg += '  Valid duration: {:.1f} s'.format(
                valid_game.ticks[-1].state.t)
            if n != 0:
                msg += '  Games: {}  Ticks: {}  Final ticks {}'.format(
                    len(games),
                    sum(len(g.ticks) for g in games),
                    sum(data['step']['n']
                        for g in games
                        for data in g.ticks[-1].bot_data))
                msg += '  Overall loss: {:.3g}'.format(
                    QBotTrainer.average_step_loss(
                        data['step']
                        for g in games
                        for t in g.ticks
                        for data in t.bot_data
                        if 'step' in data))
                msg += '  Final loss: {:.3g}'.format(
                    QBotTrainer.average_step_loss(
                        data['step']
                        for g in games
                        for data in g.ticks[-1].bot_data
                        if 'step' in data))
                if config.solo:
                    msg += '  Survival: {:.1%}'.format(
                        np.mean([g.winner is not None for g in games]))
            sys.stderr.write(msg + '\n')

            games = []

        # Train
        games.append(core.play(config, train_bots))
