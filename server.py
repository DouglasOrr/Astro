import flask
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/replay')
def replay_viewer():
    return flask.render_template('replay.html')
