import numpy as np
import sys
from . import util, core


class NothingBot:
    '''Doesn't do anything!
    '''
    def __call__(self, state):
        return 2


class ScriptBot:
    '''Tries to stay alive first, otherwise tries to shoot you.
    '''
    # these values optimized using a random walk 1-vs-1 tournament
    DEFAULT_ARGS = dict(
        avoid_distance=0.1,
        avoid_threshold=0.45,
    )

    @classmethod
    def create(cls, config, args=DEFAULT_ARGS):
        return cls(args=args, config=config)

    def __init__(self, args, config):
        self.args = args
        self.config = config

    def _fly_to(self, state, b, t, fwd):
        angle = util.norm_angle(b - state.ships.b[0])
        if angle < -t:
            return 0  # rotate left
        elif t < angle:
            return 4  # rotate right
        elif fwd:
            return 3
        else:
            return 2

    def _danger(self, x, dx, b):
        '''Am I in danger of crashing into this planet?

        x -- array(2) -- my position, relative to planet

        dx -- array(2) -- my velocity, relative to planet

        b -- float -- my bearing

        returns -- float or None -- bearing to take action, or `None`
                   if no action is required
        '''
        radius = (self.config.planet_radius + self.config.ship_radius)
        b = 2 * util.dot(util.norm(dx), x)
        c = (util.mag(x) ** 2 - (radius + self.args['avoid_distance']) ** 2)
        det = b ** 2 - 4 * c
        if 0 < det and 0 <= -b + np.sqrt(det):
            # real-valued roots & at least one positive root
            distance = -b - np.sqrt(det)
            rotation = abs(util.norm_angle(util.bearing(x) - b))
            speed = util.mag(dx)
            if distance < (speed / self.config.ship_thrust +
                           self.config.ship_rspeed / rotation) * speed:
                return util.bearing(x)
        return None

    def __call__(self, state):
        # don't crash into planets
        for i in range(state.planets.x.shape[0]):
            b = self._danger(state.ships.x[0] - state.planets.x[i],
                             state.ships.dx[0] - state.planets.dx[i],
                             state.ships.b[0])
            if b is not None:
                return self._fly_to(state, b, self.args['avoid_threshold'],
                                    fwd=True)

        # no enemy to aim for!
        if self.config.solo:
            return 2

        # aim for the enemy
        enemy_distance = util.mag(state.ships.x[1] - state.ships.x[0])
        bullet_time = (enemy_distance / self.config.bullet_speed)
        enemy_forecast = (state.ships.x[1] +
                          bullet_time *
                          (state.ships.dx[1] - state.ships.dx[0]))
        return self._fly_to(
            state,
            util.bearing(enemy_forecast - state.ships.x[0]),
            self.config.ship_radius / enemy_distance,
            fwd=False)

    @staticmethod
    def mutate(args, scale, random):
        '''Generate a local random mutation of `args`.

        args -- dict -- arguments

        scale -- float -- base scale of mutation

        random -- numpy.random.RandomState -- random generator

        returns -- dict -- mutated arguments
        '''
        args = args.copy()
        args['avoid_distance'] += scale * random.normal()
        args['avoid_threshold'] += scale * random.normal()
        return args

    @classmethod
    def optimize(cls, config, max_trials, threshold, max_mutations, scale):
        '''Optimize the arguments of this class, returning the best arguments.
        '''
        random = np.random.RandomState(config.seed)
        best = cls.DEFAULT_ARGS
        sys.stderr.write('Initial {}\n'.format(best))
        for _ in range(max_mutations):
            candidate = cls.mutate(best, scale=scale, random=random)
            winner = util.find_better(
                (core.play(cls(best, c), cls(candidate, c), c)[0]
                 for c in core.generate_configs(config)),
                max_trials=max_trials,
                threshold=threshold)
            sys.stderr.write('Trial {} -> {}\n'.format(candidate, winner))
            if winner == 1:
                best = candidate
        return best
