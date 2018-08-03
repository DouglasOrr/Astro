import numpy as np
import torch as T
import itertools as it
import functools as ft
import tensorboardX
import logging
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

    def forward(self, x):
        f = ft.reduce(
            lambda a, layer: layer(self.activation(a)),
            self.f, self.f0(x))
        v = ft.reduce(
            lambda a, layer: self.activation(layer(a)),
            self.v, self.pool(f, features=x))
        return self.v0(v)

    def evaluate(self, state):
        return self(T.autograd.Variable(
            T.FloatTensor(
                self.get_features(state))))

    def evaluate_batch(self, states):
        return self(T.autograd.Variable(
            T.FloatTensor(
                self.get_features_batch(states))))


class QNetworkTrainer:
    def __init__(self, network, writer):
        self.q = network
        self.nstep = 0
        # self._opt = T.optim.Adam(self.q.parameters(), lr=1e-3)
        # self._opt = T.optim.Rprop(self.q.parameters())
        # self._opt = T.optim.RMSprop(self.q.parameters(), eps=1e-3)
        # self._opt = T.optim.Adagrad(self.q.parameters())
        self._opt = T.optim.SGD(self.q.parameters(), lr=1e-3)
        self._writer = writer

    def step(self, inputs, actions, targets):
        inputs = T.autograd.Variable(T.FloatTensor(inputs))
        actions = T.autograd.Variable(T.LongTensor(actions))
        targets = T.autograd.Variable(T.FloatTensor(targets))

        y = self.q(inputs).gather(1, actions.view(-1, 1)).view(-1)
        losses = (targets - y) ** 2
        loss = losses.mean()
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        self._writer.add_scalar('train/loss', float(loss), self.nstep)
        self.nstep += 1
        return losses.data.numpy()


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
    __slots__ = (
        'state_f',
        'action',
        'reward',
        'discount',
        'new_state_f',
        'loss',
    )

    def __init__(self, state_f, action, reward, discount, new_state_f):
        self.state_f = state_f
        self.action = action
        self.reward = reward
        self.discount = discount
        self.new_state_f = new_state_f
        self.loss = np.inf


class QBotTrainer(QBot):
    def __init__(self, trainer, seed):
        super().__init__(trainer.q)
        self.trainer = trainer
        self.greedy = EpsilonGreedy(
            t_in=1.0,  # 1.0
            t_out=0.1,  # 0.1
            seed=seed)
        self.n_steps = 10  # 100
        self.n_ministeps = 1  # 1
        self.discount = 0.99  # 0.995
        self.n_samples = 1024  # 1024
        self.max_replay = 1000000  # 1000000
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
        xps, scored = util.partition(
            self._replay_buffer, lambda d: np.isinf(d.loss))
        assert len(xps) < self.n_samples
        if len(xps) + len(scored) < self.n_samples:
            xps += scored
        else:
            weights = np.array([d.loss for d in scored], dtype=np.float32)
            weights /= weights.sum()
            xps += [scored[i]
                    for i in self._random.choice(
                            np.arange(len(scored)),
                            size=self.n_samples - len(xps),
                            replace=False,
                            #p=weights
                    )]

        # Move nonterminals to the front, then evaluate their discounted max
        # reward using the Q function
        xps.sort(key=lambda d: d.new_state_f is None)
        targets = np.array([d.reward for d in xps])
        nonterminal = sum(1 if d.new_state_f is not None else 0 for d in xps)
        if nonterminal:
            discounts = np.array([d.discount for d in xps[:nonterminal]])
            max_rewards = T.max(
                self.q(T.autograd.Variable(T.FloatTensor(
                    self.q.to_batch(
                        [d.new_state_f for d in xps[:nonterminal]])))),
                dim=-1
            )[0].data.numpy()
            targets[:nonterminal] += discounts * max_rewards

        losses = self.trainer.step(
            inputs=self.q.to_batch([d.state_f for d in xps]),
            actions=np.array([d.action for d in xps]),
            targets=targets)

        for loss, xp in zip(losses, xps):
            xp.loss = float(loss)

        return dict(loss=losses.sum(), n=len(xps))

    def reward(self, new_state, reward):
        self._nstep_buffer[-1] += (reward,)
        if new_state is None or self.n_steps <= len(self._nstep_buffer):
            # Update the replay buffer with the new experiences.
            # We know there are only rewards for terminating states in this
            # game, so no need to store any others
            new_state_feature = (
                None
                if new_state is None else
                self.q.get_features(new_state))
            new_state_discount = 1
            discount_reward = 0
            for (feature, action, reward) in self._nstep_buffer[::-1]:
                new_state_discount *= self.discount
                discount_reward = self.discount * discount_reward + reward
                self._replay_buffer.append(Experience(
                    state_f=feature,
                    action=action,
                    reward=discount_reward,
                    discount=new_state_discount,
                    new_state_f=new_state_feature))

            # If the replay buffer is too long, throw some of it away
            if self.max_replay <= len(self._replay_buffer):
                self._replay_buffer = self._replay_buffer[self.max_replay//2:]

            # Sample from the replay buffer & take step(s) to reduce loss
            for _ in range(self.n_ministeps):
                self._step()
            self._nstep_buffer.clear()

    @property
    def data(self):
        return self._data.copy()


def train(config, interval, limit, out):
    out = util.make_counted_dir(out)
    logging.info('Logging training to: %s', out)
    writer = tensorboardX.SummaryWriter(out)

    network = ValueNetwork(solo=config.solo, nout=6)
    trainer = QNetworkTrainer(network, writer=writer)
    train_bots = [QBotTrainer(trainer, seed=n)
                  for n in range(1 if config.solo else 2)]
    eval_bots = [QBot(network)
                 for _ in range(1 if config.solo else 2)]

    total_ticks = 0
    games = []
    last_nsteps = 0
    for n, config in it.islice(enumerate(core.generate_configs(config)),
                               limit):
        if n % interval == 0:
            # Validate
            valid_game = core.play(config, eval_bots)
            core.save_log('{}/{}.log'.format(out, n), valid_game)

            msg = 'N: {}'.format(n)
            msg += '  VALID  t: {:.0f} s'.format(
                valid_game.ticks[-1].state.t)
            if n != 0:
                msg += '  TRAIN  #t: {}  #s: {}'.format(
                    sum(len(g.ticks) for g in games),
                    trainer.nstep - last_nsteps)
                msg += '  t: {:.0f} s'.format(
                    np.median([g.ticks[-1].state.t for g in games]))
                if config.solo:
                    msg += '  survive: {:.0%}'.format(
                        np.mean([g.winner is not None for g in games]))
            logging.info(msg)
            games = []
            last_nsteps = trainer.nstep

        # Train
        games.append(core.play(config, train_bots))
        total_ticks += len(games[-1].ticks)
        writer.add_scalar(
            'total/ticks', total_ticks, trainer.nstep)
        writer.add_scalar(
            'total/games', n, trainer.nstep)
        writer.add_scalar(
            'game/duration', games[-1].ticks[-1].state.t, trainer.nstep)
        if config.solo:
            writer.add_scalar(
                'game/survive', games[-1].winner is not None, trainer.nstep)
