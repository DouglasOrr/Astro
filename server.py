import flask
import astro
import uuid
import lru
import numpy as np
app = flask.Flask(__name__)

BOTS = dict(
    nothing=astro.NothingBot(),
    survival=astro.SurvivalBot(astro.SurvivalBot.DEFAULT_ARGS),
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
    config = astro.DEFAULT_CONFIG
    bot = flask.request.args['bot']
    game = dict(
        id=str(uuid.uuid4()),
        config=config,
        bot=BOTS[bot],
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
