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
    var bigger_by = 1.2;  // inflate the radius to 'match it up'
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

var player = (new function() {
    this._interval = null,
    this.play = function (config, states) {
        window.clearInterval(this._interval);
        this._interval = window.setInterval(function () {
            if (states.length === 0) {
                window.clearInterval(this._interval);
            } else {
                _render(config, states.shift());
            }
        }, config.dt * 1000);
    }
}());

function _select_replay(e) {
    if (e.target.files.length) {
        var reader = new FileReader();
        reader.onload = function (e) {
            var lines = e.target.result.trim().split('\n');
            var config = _load_json(JSON.parse(lines.shift()));
            var states = lines.map(x => _load_json(JSON.parse(x)));
            player.play(config, states);
        };
        reader.readAsText(e.target.files[0]);
        $(e.target).hide();
    }
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
    $('.replay-file').change(_select_replay);
});
