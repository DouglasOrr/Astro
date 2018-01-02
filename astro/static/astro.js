// -------------------- Rendering --------------------

function _draw_planets(ctx, config, planets) {
    ctx.fillStyle = '#68c';
    for (var i = 0; i < planets.x.length; ++i) {
        ctx.beginPath();
        ctx.arc(planets.x[i][0], planets.x[i][1], config.planet_radius,
                0, 2 * Math.PI);
        ctx.fill();
    }
}

function _draw_ships(ctx, config, ships) {
    var alpha = Math.PI / 4;
    var bigger_by = 1.1;  // inflate the radius to 'match it up'
    var line_width = 0.01;
    var colors = ['#fc0', '#d3f'];

    ctx.lineWidth = line_width;
    ctx.lineCap = 'round';
    for (var i = 0; i < ships.x.length; ++i) {
        // Compute vertices & draw
        var p = ships.x[i];
        var a0 = ships.b[i];
        var a1 = a0 + Math.PI - alpha;
        var a2 = a0 + Math.PI + alpha;
        var r = bigger_by * config.ship_radius;
        ctx.strokeStyle = colors[i];
        ctx.beginPath();
        ctx.moveTo(p[0] + r * Math.sin(a0), p[1] + r * Math.cos(a0));
        ctx.lineTo(p[0] + r * Math.sin(a1), p[1] + r * Math.cos(a1));
        ctx.lineTo(p[0] + r * Math.sin(a2), p[1] + r * Math.cos(a2));
        ctx.lineTo(p[0] + r * Math.sin(a0), p[1] + r * Math.cos(a0));
        ctx.stroke();
    }
}

function _draw_bullets(ctx, config, bullets) {
    var size = 0.01;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = size / 4;
    for (var i = 0; i < bullets.x.length; ++i) {
        var x = bullets.x[i];
        var dx = bullets.dx[i];
        var a = 0.5 * size / Math.sqrt(dx[0] ** 2 + dx[1] ** 2);
        ctx.beginPath();
        ctx.moveTo(x[0] - a * dx[0], x[1] - a * dx[1]);
        ctx.lineTo(x[0] + a * dx[0], x[1] + a * dx[1]);
        ctx.stroke();
    }
}


function _render(config, state) {
    var canvas = $('.main-canvas')[0];
    var ctx = canvas.getContext('2d');
    // The content should be centered and top aligned
    // - also flip y so that it runs up from the bottom
    var scale = Math.min(canvas.width / 2, canvas.height / 2);
    ctx.resetTransform();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(scale, -scale);

    // Clip to the dimensions of space, and fill it in
    ctx.beginPath();
    ctx.rect(-1, -1, 2, 2);
    ctx.clip();
    ctx.fillStyle = '#000';
    ctx.fill();
    ctx.lineWidth = 2 / scale;
    ctx.strokeStyle = '#a44';
    ctx.stroke();

    _draw_planets(ctx, config, state.planets);
    _draw_ships(ctx, config, state.ships);
    _draw_bullets(ctx, config, state.bullets);
}

// -------------------- Control & utility --------------------

function _load_json(obj) {
    if (obj === null) {
        return null;
    }
    var keys = Object.keys(obj);
    if (keys.length) {
        if (keys.indexOf('_shape') !== -1) {
            return obj._values;
        } else {
            delete obj._type;
            Object.keys(obj).forEach(function (k) {
                obj[k] = _load_json(obj[k]);
            });
            return obj;
        }
    } else {
        return obj;
    }
}

var replay_player = (new function() {
    this._interval = null,
    this.play = function (game) {
        window.clearInterval(this._interval);
        var frame = -1;
        this._interval = window.setInterval(function () {
            if (++frame < game.ticks.length) {
                var tick = game.ticks[frame];
                if (tick !== null) {
                    _render(game.config, tick.state);
                }
            } else {
                window.clearInterval(this._interval);
            }
        }, game.config.dt * 1000);
    }
}());

var replay = (new function() {
    this.game = null,
    this.restart = function () {
        replay_player.play(this.game);
    }
}());

function _select_replay(e) {
    if (e.target.files.length) {
        var reader = new FileReader();
        reader.onload = function (e) {
            var lines = e.target.result.trim().split('\n');
            replay.game = _load_json(JSON.parse(lines.shift()));
            replay.game.ticks = lines.map(x => _load_json(JSON.parse(x)));
            replay.restart();
        };
        reader.readAsText(e.target.files[0]);
        $(e.target).hide();
    }
}

var controller = (new function() {
    this.control = 2,
    this._up = false,
    this._down = false,
    this._left = false,
    this._right = false,
    this.keyevent = function (e) {
        if (e.key == 'ArrowUp') { e.data._up = (e.type == 'keydown'); }
        if (e.key == 'ArrowDown') { e.data._down = (e.type == 'keydown'); }
        if (e.key == 'ArrowLeft') { e.data._left = (e.type == 'keydown'); }
        if (e.key == 'ArrowRight') { e.data._right = (e.type == 'keydown'); }
        e.data.control = 2 * (1 + e.data._right - e.data._left) + (e.data._up && !e.data._down);
    }
}());

var player = (new function() {
    this._timeout = null,
    this._current_game = null,
    this.play = function (id, config, state) {
        window.clearTimeout(this._timeout);
        this._current_game = id;
        $('.game-outcome').empty();

        _render(config, state);
        var self = this;
        function tick() {
            var query = '/game/tick?' + $.param({"id": id, "control": controller.control});
            $.post(query, null, function (data) {
                if (self._current_game !== id) {
                    return; // avoid double-play
                }
                if (data.state === null) {
                    var reward = _load_json(data.reward);
                    var outcome = $('<div class="alert display-1">');
                    if (reward[0] < reward[1]) {
                        outcome.addClass('alert-danger').append('You lose 😢');
                    } else if (reward[1] < reward[0]) {
                        outcome.addClass('alert-success').append('You win ☺️');
                    } else {
                        outcome.addClass('alert-warning').append("It's a draw 😐");
                    }
                    $('.game-outcome').empty().append(outcome);
                } else {
                    _render(config, _load_json(data.state));
                    self._timeout = window.setTimeout(tick, config.dt * 1000);
                }
            });
        }
        this._timeout = window.setTimeout(tick, config.dt * 1000);
    }
}());

var game = (new function() {
    this.bot = null,
    this.restart = function() {
        var query = '/game/start?' + $.param({"bot": this.bot})
        $.post(query, null, function (data) {
            player.play(data.id, _load_json(data.config), _load_json(data.state));
        });
    }
}());

function _start_game(e) {
    game.bot = $(e.target).data('bot');
    game.restart();
    $('.bot-selector').hide();
}

function _resize_canvas() {
    var size = Math.min(window.innerWidth, window.innerHeight);
    $('.main-canvas')
	.css('left', (window.innerWidth - size) / 2 + "px")
	.css('top', (window.innerHeight - size) / 2 + "px")
	.attr('width', size)
	.attr('height', size);
}

$(function() {
    _resize_canvas();
    $(window).resize(_resize_canvas);

    // Replayer only
    if ($('.replay-file').length) {
        $('.replay-file').change(_select_replay).focus();
        $(window).keypress(function (e) {
            if (e.key == "r") {
                replay.restart();
            }
        });
    }

    // Player only
    if ($('.bot-selector').length) {
        $(window).on('keyup keydown', null, controller, controller.keyevent);
        $(window).keypress(function (e) {
            if (e.key == "r") {
                game.restart();
            }
        });
        $.get('/bots', {}, function (data) {
            $('.bot-selector')
                .empty()
                .append('<span class="lead">Choose an opponent...</span>')
                .append(data.bots.map(
                    x => $('<button class="btn btn-lg btn-outline-primary">')
                        .append(x)
                        .data('bot', x)
                        .click(_start_game)
                ));
        });
    }
});