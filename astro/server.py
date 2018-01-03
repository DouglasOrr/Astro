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
    bots = [lambda _: player_control, game['bot']]
    control = core.Bots.control(bots, game['state'])
    game['state'], reward = core.step(game['state'], control, game['config'])
    return flask.jsonify(util.to_jsonable(dict(
        id=game['id'],
        reward=reward,
        state=game['state'])))


# Views

@app.route('/')
def index():
    return flask.render_template('layout.html')
