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
    ctx.strokeStyle = '#888';
    ctx.stroke();

    _draw_planets(ctx, config, state.planets);
    _draw_ships(ctx, config, state.ships);
    _draw_bullets(ctx, config, state.bullets);
}

// -------------------- Control & utility --------------------

var renderer = (new function() {
    this._frame = null,
    this.redraw = function() {
        if (this._frame !== null) {
            var tick = this._frame.tick;
            if (tick.state !== null) {
                _render(this._frame.config, tick.state);
            }
            if (this._frame.finished) {
                var outcome = $('<div class="alert display-1">');
                if (tick.reward[0] < 0) {
                    outcome.addClass('alert-danger').append('You lose 😢');
                } else if (0 < tick.reward[0]) {
                    outcome.addClass('alert-success').append('You win ☺️');
                } else {
                    outcome.addClass('alert-warning').append("It's a draw 😐");
                }
                $('.game-outcome').empty().append(outcome);
            } else {
                $('.game-outcome').empty();
            }
        }
    },
    this.draw = function(config, tick, finished) {
        this._frame = {"config": config, "tick": tick, "finished": finished};
        this.redraw();
    }
}());

function _load_json(obj) {
    // Primitives
    if (obj === null ||
        typeof(obj) === "string" ||
        typeof(obj) === "number" ||
        typeof(obj) === "boolean") {
        return obj;
    }
    // Array
    if ($.isArray(obj)) {
        return obj.map(x => _load_json(x));
    }
    // Object, Numpy array
    var keys = Object.keys(obj);
    if (keys.indexOf('_shape') !== -1) {
        return obj._values;
    } else {
        delete obj._type;
        Object.keys(obj).forEach(function (k) {
            obj[k] = _load_json(obj[k]);
        });
        return obj;
    }
}

var replay = (new function() {
    this._interval = null,
    this.game = null,
    this.stop = function () {
        window.clearInterval(this._interval);
        this._interval = null;
    },
    this.restart = function () {
        this.stop();
        var self = this;
        var frame = -1;
        this._interval = window.setInterval(function () {
            if (++frame < self.game.ticks.length) {
                renderer.draw(
                    self.game.config,
                    self.game.ticks[frame],
                    self.game.ticks.length - 1 <= frame
                );
            } else {
                self.stop();
            }
        }, this.game.config.dt * 1000);
    }
}());

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

var game = (new function() {
    this.bot = null,
    this.stop = function() {
        this._current_game = null;
        window.clearTimeout(this._timeout);
    },
    this.restart = function() {
        var query = '/game/start?' + $.param({"bot": this.bot});
        var self = this;
        $.post(query, null, function (data) {
            var data = _load_json(data);
            self._play(data.id, data.config, data.state);
        });
    },
    this._timeout = null,
    this._current_game = null,
    this._play = function (id, config, state) {
        this.stop();
        this._current_game = id;
        renderer.draw(config, {"state": state}, false);
        var self = this;
        var last_state = null;
        function tick() {
            var query = '/game/tick?' + $.param({"id": id, "control": controller.control});
            $.post(query, null, function (data) {
                var data = _load_json(data);
                // avoid double-play
                if (self._current_game === id) {
                    var finished = (data.state === null);
                    if (finished) {
                        data.state = last_state;
                    } else {
                        last_state = data.state;
                        self._timeout = window.setTimeout(tick, config.dt * 1000);
                    }
                    // "Pretend" data is a real tick - it acts a bit like one (state & reward)
                    renderer.draw(config, data, finished);
                }
            });
        }
        this._timeout = window.setTimeout(tick, config.dt * 1000);
    }
}());

function _resize_canvas() {
    var top_pad = $('nav').height() + 15;
    var size = Math.min(window.innerWidth, window.innerHeight - top_pad);
    $('.main-canvas')
	.css('left', (window.innerWidth - size) / 2 + "px")
	.css('top', (top_pad + window.innerHeight - size) / 2 + "px")
	.attr('width', size)
	.attr('height', size);
    renderer.redraw();
}

var current = null;
function _switch_to(player) {
    if (current !== null) {
        current.stop();
    }
    current = player;
}
function _restart() {
    if (current !== null) {
        current.restart();
    }
}
function _start_game(e) {
    _switch_to(game);
    game.bot = $(e.target).data('bot');
    game.restart();
}
function _start_replay(e) {
    if (e.target.files.length) {
        var reader = new FileReader();
        reader.onload = function (e) {
            _switch_to(replay);
            var lines = e.target.result.trim().split('\n');
            replay.game = _load_json(JSON.parse(lines.shift()));
            replay.game.ticks = lines.map(x => _load_json(JSON.parse(x)));
            replay.restart();
        };
        reader.readAsText(e.target.files[0]);
    }
}

$(function() {
    _resize_canvas();
    $(window).resize(_resize_canvas);
    $(window).on('keyup keydown', null, controller, controller.keyevent);
    $('.replay-file').change(_start_replay).focus();

    // Load bot selector
    $.get('/bots', {}, function (data) {
        $('.bot-selector')
            .empty()
            .append(data.bots.map(
                x => $('<button class="dropdown-item">')
                    .append(x)
                    .data('bot', x)
                    .click(_start_game)
            ));
    });

    $(window).keypress(function (e) {
        if (e.key == "r") {
            _restart();
        }
    });

    // Player only
    // if ($('.bot-selector').length) {
    //     $(window).keypress(function (e) {
    //         if (e.key == "r") {
    //             game.restart();
    //         }
    //     });
    //     $.get('/bots', {}, function (data) {
    //         $('.bot-selector')
    //             .empty()
    //             .append('<span class="lead">Choose an opponent...</span>')
    //             .append(data.bots.map(
    //                 x => $('<button class="btn btn-lg btn-outline-primary">')
    //                     .append(x)
    //                     .data('bot', x)
    //                     .click(_start_game)
    //             ));
    //     });
    // }
});
