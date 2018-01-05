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

// -------------------- Utility --------------------


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

function _load_game(s) {
    var lines = s.trim().split('\n');
    var game = _load_json(JSON.parse(lines.shift()));
    game.ticks = lines.map(x => _load_json(JSON.parse(x)));
    return game;
}

function _set_visible(e, visible) {
    e.css('display', visible ? 'initial' : 'none');
}

// -------------------- Control & utility --------------------

var renderer = (new function() {
    this._frame = null,
    this.redraw = function() {
        if (this._frame !== null) {
            var tick = this._frame.tick;
            if (tick.state) {
                _render(this._frame.config, tick.state);
            }
            if (tick.control) {
                var data = tick.bot_data[0];
                var bot_control = tick.control[0];
                $('.arrow').each(function (i, a) {
                    var control = parseInt(a.getAttribute('data-control'));
                    if (control === bot_control) {
                        $(a).addClass('arrow-active');
                    } else {
                        $(a).removeClass('arrow-active');
                    }
                    if (data && data.q) {
                        var q = data.q[control];
                        var r = (Math.max(0, -q * 0xff) & 0xff) << 16;
                        var g = Math.max(0, q * 0xff) & 0xff;
                        $(a).css('background-color', 'rgb('+r+','+g+',0)');
                    }
                });
            }
            if ($('#show-debug-text')[0].checked &&
                tick.control && tick.reward && tick.bot_data) {
                $('.debug-text')
                    .empty()
                    .append(JSON.stringify(
                        {'control': tick.control[0],
                         'reward': tick.reward[0],
                         'bot_data': tick.bot_data[0]},
                        function (key, value) {
                            if (typeof value == 'number') {
                                return parseFloat(value.toFixed(3));
                            } else { return value; }
                        },
                        2));
            } else {
                $('.debug-text').empty();
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

var replayer = (new function() {
    this.setup = function (game) {
        this._game = game;
        $('.replayer-seek').attr('max', this._game.ticks.length - 1);
        return this;
    },
    this.restart = function () {
        return this.seek(0).play();
    },
    this.pause = function () {
        window.clearInterval(this._interval);
        this._interval = null;
        this._update_buttons();
        return this;
    },
    this.seek = function (frame) {
        if (frame < 0) {
            frame += this._game.ticks.length;
        }
        this._frame = frame;
        this._redraw();
        this._update_buttons();
        return this;
    },
    this.play = function () {
        var self = this;
        this._redraw();
        window.clearInterval(this._interval);
        this._interval = window.setInterval(function () {
            ++self._frame;
            self._redraw();
            if (self._finished()) {
                self.pause();
            }
        }, this._game.config.dt * 1000);
        this._update_buttons();
        return this;
    },
    this._finished = function () {
        return this._frame == this._game.ticks.length - 1;
    },
    this._update_buttons = function () {
        var playing = (this._interval !== null);
        var finished = this._finished();
        _set_visible($('.replayer-pause'), playing);
        _set_visible($('.replayer-play'), !playing && !finished);
        _set_visible($('.replayer-restart'), !playing && finished);
    },
    this._redraw = function() {
        $('.replayer-seek').val(this._frame);
        renderer.draw(
            this._game.config,
            this._game.ticks[this._frame],
            this._finished()
        );
    }
    this._frame = 0,
    this._interval = null,
    this._game = null
}());

var player = (new function() {
    this.setup = function (bot) {
        this._bot = bot;
        return this;
    },
    this.pause = function() {
        this._current_game = null;
        window.clearTimeout(this._timeout);
        return this;
    },
    this.restart = function() {
        var query = '/game/start?' + $.param({"bot": this._bot});
        var self = this;
        $.post(query, null, function (data) {
            var data = _load_json(data);
            self._play(data.id, data.config, data.state);
        });
        return this;
    },
    this._bot = null,
    this._timeout = null,
    this._current_game = null,
    this._play = function (id, config, state) {
        this.pause();
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

// -------------------- Wiring things up --------------------

function _resize_canvas() {
    var width = $('.main-canvas-holder').innerWidth();
    var height = window.innerHeight - $('.main-canvas-holder').offset().top;
    var size = Math.min(width, height);
    $('.main-canvas')
	.css('left', (width - size) / 2 + "px")
	.css('top', (height - size) / 2 + "px")
	.attr('width', size)
	.attr('height', size);
    renderer.redraw();
}

var current = {'restart': function() { },
               'pause': function() { }};
function _switch_to(player) {
    current.pause();
    current = player;
    $('.sidebar').css('visibility', Object.is(current, replayer) ? 'visible' : 'hidden');
    return current;
}
function _start_game(e) {
    _switch_to(player)
        .setup($(e.target).data('bot'))
        .restart();
}
function _start_replay(e) {
    if (e.target.files.length) {
        var reader = new FileReader();
        reader.onload = function (e) {
            _switch_to(replayer)
                .setup(_load_game(e.target.result))
                .restart();
        };
        reader.readAsText(e.target.files[0]);
    }
}

$(function() {
    _resize_canvas();
    $(window).resize(_resize_canvas);

    // Keyboard control
    $(window).on('keyup keydown', null, controller, controller.keyevent);
    $(window).keypress(function (e) {
        if (e.key == 'r') {
            current.restart();
        } else if (e.key == 'o') {
            $('.replay-file').click();
        }
    });

    // Replayer control
    $('.replayer-play').on('click', function (e) {
	replayer.play();
    });
    $('.replayer-pause').on('click', function (e) {
	replayer.pause();
    });
    $('.replayer-restart').on('click', function (e) {
	replayer.seek(0).play();
    });
    $('.replayer-start').on('click', function (e) {
	replayer.pause().seek(0);
    });
    $('.replayer-end').on('click', function (e) {
	replayer.pause().seek(-1);
    });
    $('.replayer-seek').on('input', function (e) {
	replayer.pause().seek(parseInt($(e.target).val()));
    });
    $('#show-debug-text').on('change', function (e) {
        renderer.redraw();
    });

    // Replay selector
    $('.replay-file').change(_start_replay).focus();

    // Bot selector
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
});
