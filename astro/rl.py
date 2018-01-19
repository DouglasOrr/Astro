import numpy as np
import torch as T
import itertools as it
import functools as ft
import sys
from . import util, core


class EpsilonGreedy:
    """A stateful bot which returns an action according to a random policy, or
    `None` if the random policy should not be active.
    """
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
    """Deep Q learning RL network for playing astro.
    """
    @staticmethod
    def get_features_shape(state):
        return (
            state.planets.x.shape[0] +
            state.bullets.x.shape[0],
            1 + 4 + 5 * state.ships.x.shape[0])

    @classmethod
    def get_features(cls, state):
        """Create the feature vector from the current state.
        A little feature engineering here to make the model's job easier
        - concatenate the ships feature vector onto every other object.

        state -- astro.State

        returns -- array(N x D) -- feature array (floats)
        """
        def feature(bodies):
            return np.concatenate(
                (bodies.x, bodies.dx) +
                (()
                 if bodies.b is None else
                 (util.norm_angle(bodies.b[:, np.newaxis]) / np.pi,)),
                axis=1)

        # dim 0: [planets, bullets]
        # dim 1: [type, ships, object]
        features = np.zeros(cls.get_features_shape(state),
                            dtype=np.float32)
        ibullets = state.planets.x.shape[0]
        features[:ibullets, 0] = 0  # planet flag
        features[ibullets:, 0] = 1  # bullet flag
        iobject = 1 + 5 * state.ships.x.shape[0]
        features[:, 1:iobject] = feature(state.ships).flatten()
        features[:ibullets, iobject:] = feature(state.planets)
        features[ibullets:, iobject:] = feature(state.bullets)

        return features

    @staticmethod
    def to_batch(features):
        """Put a list of features into a batch.

        states -- [array(N_f x D)] -- B features (from `get_features`)

        returns -- array(B x N x D) -- batched feature array;
                   note that result[:, :, 0] - which is the feature
                   "object type" should be used as a mask - all features with
                   (result[:, :, 0] < 0) should be skipped
        """
        if any(f.shape[1] != features[0].shape[1] for f in features):
            raise ValueError(
                'Feature dimensions do not match - cannot mix solo & nonsolo'
                ' games in a single batch')

        # Use -1 to pad any remaining space at the end of each item
        result = np.full(
            (len(features),
             max(f.shape[0] for f in features),
             features[0].shape[1]),
            fill_value=-1,
            dtype=np.float32)
        for b, feature in zip(it.count(), features):
            result[b, 0:feature.shape[0], :] = feature
        return result

    @classmethod
    def get_features_batch(cls, states):
        """As `get_features`, but for a batch of states.

        states -- [astro.State] -- B states

        returns -- array(B x N x D) -- batched feature array;
                   note that result[:, :, 0] - which is the feature
                   "object type" should be used as a mask - all features with
                   (result[:, :, 0] < 0) should be skipped
        """
        return cls.to_batch([cls.get_features(s) for s in states])

    @staticmethod
    def masked_max(x, features):
        """Max-pooling over axis -2, with masking according to the
        "object type" field (features[..., 0] < 0).

        x -- Variable[B x N x X] -- processed features

        features -- Variable[B x N x F] -- original feature values for masking

        returns -- Variable[B x X] -- masked & pooled
        """
        # Torch doesn't like np.inf here, so just use a large value
        xm = x - 1e9 * (
            features[..., 0] < 0)[..., np.newaxis].type(x.data.type())
        return T.max(xm, -2)[0]

    @staticmethod
    def masked_sum(x, features):
        """As masked_max (but performs sum-pooling).
        """
        xm = x * (0 <= features[..., 0])[..., np.newaxis].type(x.data.type())
        return T.sum(xm, -2)

    def __init__(self, solo, nout):
        super().__init__()
        nf = (10 if solo else 15)
        nh, d = 32, 2
        self.activation = T.nn.functional.softsign
        self.f0 = T.nn.Linear(nf, nh)
        self.f = T.nn.ModuleList([T.nn.Linear(nh, nh) for _ in range(d)])
        self.pool = self.masked_max
        self.v = T.nn.ModuleList([T.nn.Linear(nh, nh) for _ in range(d)])
        self.v0 = T.nn.Linear(nh, nout)

        # self.opt = T.optim.Adam(self.parameters(), lr=1e-3)
        # self.opt = T.optim.Rprop(self.parameters())
        # self.opt = T.optim.RMSprop(self.parameters(), eps=1e-3)
        # self.opt = T.optim.Adagrad(self.parameters())
        self.opt = T.optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, x):
        f = ft.reduce(
            lambda a, layer: layer(self.activation(a)),
            self.f, self.f0(x))
        v = ft.reduce(
            lambda a, layer: self.activation(layer(a)),
            self.v, self.pool(f, features=x))
        return T.tanh(self.v0(v))

    def evaluate(self, state):
        return self(T.autograd.Variable(
            T.FloatTensor(
                self.get_features(state))))

    def evaluate_batch(self, states):
        return self(T.autograd.Variable(
            T.FloatTensor(
                self.get_features_batch(states))))


class QBot(core.Bot):
    """A basic "evaluation mode" Q-learning bot with no epsilon-greedy
    policy & no training.
    """
    def __init__(self, network):
        self.q = network
        self._last_q = None

    def __call__(self, state):
        self._last_q = self.q.evaluate(state).data.numpy()
        return np.argmax(self._last_q)

    @property
    def data(self):
        return dict(q=self._last_q)


class Experience:
    __slots__ = ('feature', 'action', 'reward', 'loss')

    def __init__(self, feature, action, reward):
        self.feature = feature
        self.action = action
        self.reward = reward
        self.loss = np.inf


class QBotTrainer(QBot):
    def __init__(self, network, seed):
        super().__init__(network)
        self.greedy = EpsilonGreedy(t_in=1.0, t_out=0.1, seed=seed)
        self.n_steps = 10
        self.discount = 0.99
        self.n_samples = 128
        self.max_replay = 100000
        self._nstep_buffer = []
        self._replay_buffer = []
        self._data = dict(greedy=None, q=None)
        self._random = np.random.RandomState(seed)

    def __call__(self, state):
        # Act according to an e-greedy policy or Q
        f = self.q.get_features(state)
        q = self.q(T.autograd.Variable(T.FloatTensor(f))).data.numpy()
        greedy = self.greedy(state)
        action = int(greedy if greedy is not None else np.argmax(q))
        self._nstep_buffer.append((f, action))
        self._data['greedy'] = greedy
        self._data['q'] = q
        return action

    def _step(self):
        """Sample from the replay buffer, and take an optimization step.
        """
        assert self.n_steps < self.n_samples
        if len(self._replay_buffer) <= self.n_samples:
            xps = self._replay_buffer
        else:
            xps = self._replay_buffer[-self.n_steps:]
            sbuf = self._replay_buffer[:-self.n_steps]
            weights = np.array([d.loss for d in sbuf], dtype=np.float32)
            weights /= weights.sum()
            for i in self._random.choice(
                    np.arange(len(sbuf)),
                    size=self.n_samples - self.n_steps,
                    replace=False,
                    p=weights):
                xps.append(sbuf[i])

        inputs = T.autograd.Variable(T.FloatTensor(ValueNetwork.to_batch(
            [d.feature for d in xps])))
        actions = T.autograd.Variable(T.LongTensor([d.action for d in xps]))
        targets = T.autograd.Variable(T.FloatTensor([d.reward for d in xps]))

        y = self.q(inputs).gather(1, actions.view(-1, 1)).view(-1)
        losses = (targets - y) ** 2
        loss = losses.sum()
        self.q.opt.zero_grad()
        (loss / targets.nelement()).backward()
        self.q.opt.step()

        for l, xp in zip(losses, xps):
            xp.loss = float(l)
        return dict(loss=float(loss), n=targets.nelement())

    def reward(self, new_state, reward):
        if new_state is None or self.n_steps <= len(self._nstep_buffer):
            # Update the replay buffer with the new (state, action, reward)s
            # we know there are only rewards for terminating states, in this
            # game, so no need to store any others
            discounted_reward = (
                reward
                if new_state is None else
                self.discount * float(T.max(self.q.evaluate(new_state))))
            for n, (feature, action) in enumerate(self._nstep_buffer):
                p = len(self._nstep_buffer) - 1 - n
                r = discounted_reward * (self.discount ** p)
                self._replay_buffer.append(Experience(feature, action, r))

            # If the replay buffer is too long, throw some of it away
            if self.max_replay <= len(self._replay_buffer):
                self._replay_buffer = self._replay_buffer[self.max_replay//2:]

            # Sample from the replay buffer & take a step to reduce loss
            self._data['step'] = self._step()

            self._nstep_buffer.clear()
        else:
            if 'step' in self._data:
                del self._data['step']

    @property
    def data(self):
        return self._data.copy()

    @staticmethod
    def average_step_loss(steps):
        """Compute the average (weighted) loss from a sequence of steps.
        """
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
            msg += '  VALID  t: {:.0f} s'.format(
                valid_game.ticks[-1].state.t)
            if n != 0:
                msg += '  TRAIN  #t: {}  #s: {}'.format(
                    sum(len(g.ticks) for g in games),
                    sum(1 if 'step' in data else 0
                        for g in games
                        for t in g.ticks
                        for data in t.bot_data))
                msg += '  L: {:.3g}  L_final: {:.3g}'.format(
                    QBotTrainer.average_step_loss(
                        data['step']
                        for g in games
                        for t in g.ticks
                        for data in t.bot_data
                        if 'step' in data),
                    QBotTrainer.average_step_loss(
                        data['step']
                        for g in games
                        for data in g.ticks[-1].bot_data
                        if 'step' in data))
                msg += '  t: {:.0f} s'.format(
                    np.median([g.ticks[-1].state.t for g in games]))
                if config.solo:
                    msg += '  live: {:.0%}'.format(
                        np.mean([g.winner is not None for g in games]))
            sys.stderr.write(msg + '\n')

            games = []

        # Train
        games.append(core.play(config, train_bots))
