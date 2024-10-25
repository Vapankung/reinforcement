const canvas = document.getElementById('mazeCanvas');
const ctx = canvas.getContext('2d');
const solveButton = document.getElementById('solveButton');

const mazeData = [
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
];

const cellSize = canvas.width / mazeData.length;

function drawMaze() {
    for (let y = 0; y < mazeData.length; y++) {
        for (let x = 0; x < mazeData[y].length; x++) {
            if (mazeData[y][x] === 1) {
                ctx.fillStyle = '#000';
            } else {
                ctx.fillStyle = '#fff';
            }
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }
}

function drawAgent(position) {
    ctx.fillStyle = 'blue';
    ctx.beginPath();
    ctx.arc(
        position.x * cellSize + cellSize / 2,
        position.y * cellSize + cellSize / 2,
        cellSize / 3,
        0,
        2 * Math.PI
    );
    ctx.fill();
}

async function solveMaze() {
    const response = await fetch('/solve_maze');
    const data = await response.json();
    const path = data.path;
    let index = 0;

    function animate() {
        if (index < path.length) {
            drawMaze();
            drawAgent(path[index]);
            index++;
            setTimeout(animate, 500);
        }
    }

    animate();
}

solveButton.addEventListener('click', () => {
    drawMaze();
    solveMaze();
});

drawMaze();
