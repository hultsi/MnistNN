const canvas = document.getElementById('myCanvas');
const body = document.getElementsByTagName("body")[0];
const ctx = canvas.getContext('2d');
const currentColor = "rgb(255,255,255)";
const currentBg = "rgb(0,0,0)";

let isMouseDown = false;
let currentSize = 28;

canvas.width = 280;
canvas.height = 280;

const clearCanvas = function() {
    canvas.style.zIndex = 8;
    ctx.fillStyle = currentBg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

const getMousePos = function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

const onMouseDown = function onMouseDown(evt, canvas) {
    isMouseDown = true;
    var currentPosition = getMousePos(canvas, evt);
    ctx.moveTo(currentPosition.x, currentPosition.y);
    ctx.beginPath();
    ctx.lineWidth  = currentSize;
    ctx.lineCap = "round";
    ctx.strokeStyle = currentColor;
}

const onMouseUp = function onMouseUp() {
    isMouseDown = false;
}

const onMouseMove = function onMouseMove(evt, canvas) {
    if (isMouseDown) {
        var currentPosition = getMousePos(canvas, evt);
        ctx.lineTo(currentPosition.x, currentPosition.y)
        ctx.stroke();
    }
}

document.getElementById('clearCache').addEventListener('click', clearCanvas);
canvas.addEventListener('mousedown', (e) => onMouseDown(e, canvas));
canvas.addEventListener('mousemove', (e) => onMouseMove(e, canvas));
canvas.addEventListener('mouseup', onMouseUp);

clearCanvas();