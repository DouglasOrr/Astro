import numpy as np
import torch as T
import itertools as it
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


class QNetwork(T.nn.Module):
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

    def __init__(self, solo):
        super().__init__()
        nf, nq = (10 if solo else 15), 6
        nh = 32
        self.activation = T.nn.functional.elu
        self.f0 = T.nn.Linear(nf, nh)
        self.f1 = T.nn.Linear(nh, nh)
        self.pool = lambda x: T.max(x, 0)[0]
        self.q1 = T.nn.Linear(nh, nh)
        self.q0 = T.nn.Linear(nh, nq)

        self.opt = T.optim.SGD(self.parameters(), lr=1e-6)

    def __call__(self, x):
        # return T.tanh(self.q0(self.pool(self.f0(x))))
        pool = self.pool(self.f1(self.activation(self.f0(x))))
        return T.tanh(self.q0(self.activation(self.q1(pool))))


class QBot(core.Bot):
    '''A basic "evaluation mode" Q-learning bot with no epsilon-greedy
    policy & no training.
    '''
    def __init__(self, network):
        self.network = network
        self._last_q = None

    def q(self, state):
        return self.network(
            T.autograd.Variable(
                T.FloatTensor(
                    self.network.get_features(state))))

    def __call__(self, state):
        self._last_q = self.q(state).data.numpy()
        return np.argmax(self._last_q)

    @property
    def data(self):
        return dict(q=self._last_q)


class QBotTrainer(QBot):
    def __init__(self, network, seed):
        super().__init__(network)
        self.greedy = EpsilonGreedy(t_in=1.0, t_out=0.1, seed=seed)
        self.n_steps = 8
        self.discount = 0.98
        self._buffer = []
        self._data = dict(step=None, greedy=False, q=None)

    def __call__(self, state):
        # Act according to an e-greedy policy or Q
        q = self.q(state)
        greedy = self.greedy(state)
        action = (greedy if greedy is not None else np.argmax(q.data.numpy()))
        self._buffer.append(q[action])
        self._data['greedy'] = greedy is not None
        self._data['q'] = q
        return action

    def reward(self, new_state, reward):
        if new_state is None or self.n_steps <= len(self._buffer):
            # we know there are only rewards for terminating states, in this
            # game, so no need to store any others
            if new_state is None:
                r = reward
            else:
                r = self.discount * np.max(self.q(new_state).data.numpy())
            target = T.autograd.Variable(T.FloatTensor(
                r * (self.discount ** np.arange(len(self._buffer)))))
            q = T.cat(self._buffer[::-1])
            e = ((target - q) ** 2).sum(0)
            self.network.opt.zero_grad()
            (e / self.n_steps).backward()
            self.network.opt.step()
            self._data['step'] = dict(loss=float(e.data), n=len(self._buffer))
            self._buffer.clear()
        else:
            self._data['step'] = None

    @property
    def data(self):
        return self._data.copy()

    @staticmethod
    def average_step_loss(steps):
        '''Compute the average (weighted) loss from a sequence of steps.
        '''
        loss, n = 0, 0
        for step in steps:
            if step is not None:
                loss += step['loss']
                n += step['n']
        return loss / n


def train(config, interval, limit, log_prefix=None):
    network = QNetwork(solo=config.solo)
    train_bots = [QBotTrainer(network, n)
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
                msg += '  Overall loss: {:.3g}'.format(
                    QBotTrainer.average_step_loss(
                        data['step']
                        for g in games
                        for t in g.ticks
                        for data in t.bot_data))
                msg += '  Final loss: {:.3g}'.format(
                    QBotTrainer.average_step_loss(
                        data['step']
                        for g in games
                        for data in g.ticks[-1].bot_data))
                if config.solo:
                    msg += '  Survival: {:.1%}'.format(
                        np.mean([g.winner is not None for g in games]))
            sys.stderr.write(msg + '\n')

            games = []

        # Train
        games.append(core.play(config, train_bots))
