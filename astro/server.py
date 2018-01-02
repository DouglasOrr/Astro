import flask
import uuid
import lru
import numpy as np
import random
from . import core, util, script


app = flask.Flask(__name__)

BOTS = dict(
    nothing=(lambda config: script.NothingBot()),
    script=(lambda config: script.ScriptBot.create(config)),
)
GAMES = lru.LRU(10)  # to stop us running out of memory


def _render_game(game):
    return flask.jsonify(dict(
        id=game['id'],
        bot=game['bot']['name'],
        config=util.to_jsonable(game['config']),
        state=util.to_jsonable(game['state']),
        reward=util.to_jsonable(game['reward'])))


# App

@app.route('/bots')
def bots():
    return flask.jsonify(dict(bots=list(BOTS.keys())))


@app.route('/game/start', methods=['POST'])
def game_start():
    config = core.DEFAULT_CONFIG._replace(seed=random.randint(0, 1 << 30))
    bot = flask.request.args['bot']
    game = dict(
        id=str(uuid.uuid4()),
        config=config,
        bot=BOTS[bot](config),
        state=core.create(config))
    GAMES[game['id']] = game
    return flask.jsonify(dict(
        id=game['id'],
        bot=bot,
        config=util.to_jsonable(game['config']),
        state=util.to_jsonable(game['state'])))


@app.route('/game/tick', methods=['POST'])
def game_tick():
    game = GAMES[flask.request.args['id']]
    player_control = int(flask.request.args['control'])
    bot_control = game['bot'](core.swap_ships(game['state']))
    game['state'], reward = core.step(
        game['state'],
        np.array([player_control, bot_control]),
        game['config'])
    return flask.jsonify(dict(
        id=game['id'],
        state=util.to_jsonable(game['state']),
        reward=util.to_jsonable(reward)))


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
