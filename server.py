import flask
import astro
import uuid
import lru
import numpy as np
import random
app = flask.Flask(__name__)

BOTS = dict(
    nothing=(lambda config: astro.NothingBot()),
    script=(lambda config: astro.ScriptBot(
        astro.ScriptBot.DEFAULT_ARGS, config)),
)
GAMES = lru.LRU(10)  # to stop us running out of memory


def _render_game(game):
    return flask.jsonify(dict(
        id=game['id'],
        bot=game['bot']['name'],
        config=astro._to_json(game['config']),
        state=astro._to_json(game['state']),
        reward=astro._to_json(game['reward'])))


# App

@app.route('/bots')
def bots():
    return flask.jsonify(dict(bots=list(BOTS.keys())))


@app.route('/game/start', methods=['POST'])
def game_start():
    config = astro.DEFAULT_CONFIG._replace(seed=random.randint(0, 1 << 30))
    bot = flask.request.args['bot']
    game = dict(
        id=str(uuid.uuid4()),
        config=config,
        bot=BOTS[bot](config),
        state=astro.create(config))
    GAMES[game['id']] = game
    return flask.jsonify(dict(
        id=game['id'],
        bot=bot,
        config=astro._to_json(game['config']),
        state=astro._to_json(game['state'])))


@app.route('/game/tick', methods=['POST'])
def game_tick():
    game = GAMES[flask.request.args['id']]
    player_control = int(flask.request.args['control'])
    bot_control = game['bot'](astro.swap_ships(game['state']))
    game['state'], reward = astro.step(
        game['state'],
        np.array([player_control, bot_control]),
        game['config'])
    return flask.jsonify(dict(
        id=game['id'],
        state=astro._to_json(game['state']),
        reward=astro._to_json(reward)))


# Views

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/player')
def player():
    return flask.render_template('player.html')


@app.route('/replayer')
def replayer():
    return flask.render_template('replay.html')
