class CavaloSolitario {
    constructor() {
        this.boardSize = 8;
        this.tourType = 'open';
        this.board = [];
        this.currentPosition = null;
        this.visited = new Set();
        this.moveHistory = [];
        this.startTime = null;
        this.timer = null;
        this.moveCount = 0;
        this.gameActive = false;
        this.leaderboard = this.loadLeaderboard();
        
        this.initializeElements();
        this.bindEvents();
        this.updateLeaderboard();
    }

    initializeElements() {
        this.boardElement = document.getElementById('game-board');
        this.timerElement = document.getElementById('timer');
        this.movesElement = document.getElementById('moves');
        this.visitedElement = document.getElementById('visited');
        this.boardSizeSelect = document.getElementById('board-size');
        this.tourTypeSelect = document.getElementById('tour-type');
        this.newGameBtn = document.getElementById('new-game-btn');
        this.undoBtn = document.getElementById('undo-btn');
        this.hintBtn = document.getElementById('hint-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.aiHintBtn = document.getElementById('ai-hint-btn');
        this.victoryModal = document.getElementById('victory-modal');
        this.gameoverModal = document.getElementById('gameover-modal');
        this.leaderboardContent = document.getElementById('leaderboard-content');
    }

    bindEvents() {
        this.newGameBtn.addEventListener('click', () => this.startNewGame());
        this.undoBtn.addEventListener('click', () => this.undoMove());
        this.hintBtn.addEventListener('click', () => this.showHint());
        this.resetBtn.addEventListener('click', () => this.resetGame());
        this.aiHintBtn.addEventListener('click', () => this.getAIHint());
        
        // Modal events
        document.getElementById('play-again-btn').addEventListener('click', () => {
            this.closeModal(this.victoryModal);
            this.startNewGame();
        });
        document.getElementById('close-modal-btn').addEventListener('click', () => {
            this.closeModal(this.victoryModal);
        });
        document.getElementById('try-again-btn').addEventListener('click', () => {
            this.closeModal(this.gameoverModal);
            this.resetGame();
        });
        document.getElementById('close-gameover-btn').addEventListener('click', () => {
            this.closeModal(this.gameoverModal);
        });

        // Close modals when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target === this.victoryModal) this.closeModal(this.victoryModal);
            if (e.target === this.gameoverModal) this.closeModal(this.gameoverModal);
        });
    }

    startNewGame() {
        this.boardSize = parseInt(this.boardSizeSelect.value);
        this.tourType = this.tourTypeSelect.value;
        this.gameActive = true;
        this.moveCount = 0;
        this.visited.clear();
        this.moveHistory = [];
        this.currentPosition = null;
        
        this.createBoard();
        this.updateStats();
        this.disableGameButtons();
        this.startTimer();
    }

    createBoard() {
        this.board = [];
        this.boardElement.innerHTML = '';
        this.boardElement.style.gridTemplateColumns = `repeat(${this.boardSize}, 1fr)`;
        // Ajuste dinâmico do tamanho das células para tabuleiros maiores
        const cellSize = Math.max(30, 50 - (this.boardSize - 8) * 5);
        this.boardElement.style.setProperty('--cell-size', `${cellSize}px`);

        for (let row = 0; row < this.boardSize; row++) {
            this.board[row] = [];
            for (let col = 0; col < this.boardSize; col++) {
                const cell = document.createElement('div');
                cell.className = `board-cell ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
                cell.dataset.row = row;
                cell.dataset.col = col;
                
                cell.addEventListener('click', () => this.handleCellClick(row, col));
                
                this.board[row][col] = cell;
                this.boardElement.appendChild(cell);
            }
        }
    }

    handleCellClick(row, col) {
        if (!this.gameActive) return;

        if (this.currentPosition === null) {
            // First move - place the knight
            this.placeKnight(row, col);
        } else {
            // Check if the clicked position is a valid move
            const possibleMoves = this.getPossibleMoves(this.currentPosition.row, this.currentPosition.col);
            const clickedPos = { row, col };
            
            if (possibleMoves.some(move => move.row === clickedPos.row && move.col === clickedPos.col)) {
                this.moveKnight(row, col);
            }
        }
    }

    placeKnight(row, col) {
        this.currentPosition = { row, col };
        this.visited.add(`${row},${col}`);
        this.moveHistory.push({ row, col, moveNumber: 1 });
        
        const cell = this.board[row][col];
        cell.classList.add('current');
        cell.innerHTML = '<i class="fas fa-chess-knight"></i>';
        
        this.highlightPossibleMoves();
        this.updateStats();
        this.enableGameButtons();
    }

    moveKnight(row, col) {
        const oldRow = this.currentPosition.row;
        const oldCol = this.currentPosition.col;
        
        // Update previous position
        const oldCell = this.board[oldRow][oldCol];
        oldCell.classList.remove('current');
        oldCell.classList.add('visited');
        oldCell.innerHTML = this.moveHistory.length;
        
        // Update new position
        this.currentPosition = { row, col };
        this.visited.add(`${row},${col}`);
        this.moveCount++;
        this.moveHistory.push({ row, col, moveNumber: this.moveCount + 1 });
        
        const newCell = this.board[row][col];
        newCell.classList.add('current');
        newCell.innerHTML = '<i class="fas fa-chess-knight"></i>';
        
        this.highlightPossibleMoves();
        this.updateStats();
        
        // Check win condition
        if (this.checkWinCondition()) {
            this.handleVictory();
        } else if (this.getPossibleMoves(row, col).length === 0) {
            this.handleGameOver();
        }
    }

    getPossibleMoves(row, col) {
        const moves = [];
        const knightMoves = [
            [-2, -1], [-2, 1], [-1, -2], [-1, 2],
            [1, -2], [1, 2], [2, -1], [2, 1]
        ];
        
        for (const [dRow, dCol] of knightMoves) {
            const newRow = row + dRow;
            const newCol = col + dCol;
            
            if (this.isValidPosition(newRow, newCol) && !this.visited.has(`${newRow},${newCol}`)) {
                moves.push({ row: newRow, col: newCol });
            }
        }
        
        return moves;
    }

    isValidPosition(row, col) {
        return row >= 0 && row < this.boardSize && col >= 0 && col < this.boardSize;
    }

    highlightPossibleMoves() {
        // Clear previous highlights
        this.board.forEach(row => {
            row.forEach(cell => {
                cell.classList.remove('possible-move', 'hint');
            });
        });
        
        if (this.currentPosition) {
            const possibleMoves = this.getPossibleMoves(this.currentPosition.row, this.currentPosition.col);
            possibleMoves.forEach(move => {
                this.board[move.row][move.col].classList.add('possible-move');
            });
        }
    }

    showHint() {
        if (!this.currentPosition) return;
        
        const possibleMoves = this.getPossibleMoves(this.currentPosition.row, this.currentPosition.col);
        if (possibleMoves.length === 0) return;
        
        // Clear previous hints
        this.board.forEach(row => {
            row.forEach(cell => {
                cell.classList.remove('hint');
            });
        });
        
        // Show hint for the best move (using Warnsdorff's algorithm)
        const bestMove = this.getBestMove(possibleMoves);
        if (bestMove) {
            this.board[bestMove.row][bestMove.col].classList.add('hint');
        }
    }

    getBestMove(possibleMoves) {
        // Warnsdorff's algorithm: choose the move with the fewest possible next moves
        let bestMove = null;
        let minNextMoves = Infinity;
        
        for (const move of possibleMoves) {
            const nextMoves = this.getPossibleMoves(move.row, move.col);
            if (nextMoves.length < minNextMoves) {
                minNextMoves = nextMoves.length;
                bestMove = move;
            }
        }
        
        return bestMove;
    }

    checkWinCondition() {
        const totalCells = this.boardSize * this.boardSize;
        const visitedCount = this.visited.size;
        
        if (this.tourType === 'closed') {
            // For closed tour, we need to visit all cells AND be able to return to start
            if (visitedCount === totalCells) {
                const startPos = this.moveHistory[0];
                const possibleMoves = this.getPossibleMoves(this.currentPosition.row, this.currentPosition.col);
                return possibleMoves.some(move => move.row === startPos.row && move.col === startPos.col);
            }
        } else {
            // For open tour, just visit all cells
            return visitedCount === totalCells;
        }
        
        return false;
    }

    handleVictory() {
        this.gameActive = false;
        this.stopTimer();
        
        const timeElapsed = this.getTimeElapsed();
        const timeString = this.formatTime(timeElapsed);
        
        // Update victory modal
        document.getElementById('victory-time').textContent = timeString;
        document.getElementById('victory-moves').textContent = this.moveCount;
        document.getElementById('victory-type').textContent = 
            this.tourType === 'closed' ? 'Passeio Fechado' : 'Passeio Aberto';
        
        // Save to leaderboard
        this.saveToLeaderboard(timeElapsed, this.moveCount);
        
        this.showModal(this.victoryModal);
    }

    handleGameOver() {
        this.gameActive = false;
        this.stopTimer();
        this.showModal(this.gameoverModal);
    }

    undoMove() {
        if (this.moveHistory.length <= 1) return;
        
        // Pega a posição atual, que será removida
        const undoneMove = this.moveHistory.pop();
        const undoneCell = this.board[undoneMove.row][undoneMove.col];
        
        // Limpa a casa que foi desfeita
        undoneCell.classList.remove('current');
        undoneCell.innerHTML = '';
        this.visited.delete(`${undoneMove.row},${undoneMove.col}`);
        
        // Define a nova posição atual a partir do histórico
        const lastMove = this.moveHistory[this.moveHistory.length - 1];
        this.currentPosition = { row: lastMove.row, col: lastMove.col };
        
        // Atualiza a nova casa atual
        const newCurrentCell = this.board[this.currentPosition.row][this.currentPosition.col];
        newCurrentCell.classList.remove('visited');
        newCurrentCell.classList.add('current');
        newCurrentCell.innerHTML = '<i class="fas fa-chess-knight"></i>';
        
        // Atualiza a interface
        this.highlightPossibleMoves();
        this.updateStats();
    }

    resetGame() {
        this.stopTimer();
        this.startNewGame();
    }

    startTimer() {
        this.startTime = Date.now();
        this.timer = setInterval(() => {
            this.updateTimer();
        }, 1000);
    }

    stopTimer() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }

    updateTimer() {
        const timeElapsed = this.getTimeElapsed();
        this.timerElement.textContent = this.formatTime(timeElapsed);
    }

    getTimeElapsed() {
        return this.startTime ? Date.now() - this.startTime : 0;
    }

    formatTime(milliseconds) {
        const seconds = Math.floor(milliseconds / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    updateStats() {
        this.movesElement.textContent = this.moveCount;
        this.visitedElement.textContent = this.visited.size;
    }

    enableGameButtons() {
        this.undoBtn.disabled = false;
        this.hintBtn.disabled = false;
        this.resetBtn.disabled = false;
        if (this.boardSize === 5) {
            this.aiHintBtn.disabled = false;
        }
    }

    disableGameButtons() {
        this.undoBtn.disabled = true;
        this.hintBtn.disabled = true;
        this.resetBtn.disabled = true;
        this.aiHintBtn.disabled = true;
    }

    showModal(modal) {
        modal.style.display = 'block';
    }

    closeModal(modal) {
        modal.style.display = 'none';
    }

    loadLeaderboard() {
        const saved = localStorage.getItem('cavaloSolitarioLeaderboard');
        return saved ? JSON.parse(saved) : {};
    }

    saveLeaderboard() {
        localStorage.setItem('cavaloSolitarioLeaderboard', JSON.stringify(this.leaderboard));
    }

    saveToLeaderboard(timeElapsed, moves) {
        const key = `${this.boardSize}x${this.boardSize}_${this.tourType}`;
        
        if (!this.leaderboard[key]) {
            this.leaderboard[key] = [];
        }
        
        const entry = {
            time: timeElapsed,
            moves: moves,
            date: new Date().toISOString(),
            timeString: this.formatTime(timeElapsed)
        };
        
        this.leaderboard[key].push(entry);
        
        // Keep only top 5 scores
        this.leaderboard[key].sort((a, b) => a.time - b.time || a.moves - b.moves);
        this.leaderboard[key] = this.leaderboard[key].slice(0, 5);
        
        this.saveLeaderboard();
        this.updateLeaderboard();
    }

    updateLeaderboard() {
        this.leaderboardContent.innerHTML = '';
        
        const allScores = [];
        
        for (const key in this.leaderboard) {
            if (this.leaderboard[key].length > 0) {
                const bestScore = this.leaderboard[key][0];
                const [size, type] = key.split('_');
                allScores.push({ ...bestScore, size, type });
            }
        }

        // Ordena para mostrar os melhores scores primeiro (menor tempo)
        allScores.sort((a, b) => a.time - b.time);
        
        if (allScores.length === 0) {
            this.leaderboardContent.innerHTML = '<p style="text-align: center; color: #666;">Nenhum recorde ainda. Complete um jogo para aparecer aqui!</p>';
            return;
        }

        allScores.forEach(score => {
            const item = document.createElement('div');
            item.className = 'leaderboard-item best';
            
            item.innerHTML = `
                <div class="leaderboard-info">
                    <div class="leaderboard-size">${score.size}</div>
                    <div class="leaderboard-type">${score.type === 'closed' ? 'Passeio Fechado' : 'Passeio Aberto'}</div>
                </div>
                <div class="leaderboard-stats">
                    <div class="leaderboard-time">${score.timeString}</div>
                    <div class="leaderboard-moves">${score.moves} jogadas</div>
                </div>
            `;
            
            this.leaderboardContent.appendChild(item);
        });
    }

    async getAIHint() {
        if (!this.currentPosition || this.boardSize !== 5) {
            alert("A Dica da IA só está disponível para o tabuleiro 5x5.");
            return;
        }

        // Limpa dicas anteriores
        this.board.forEach(row => row.forEach(cell => cell.classList.remove('ai-hint')));

        // 1. Constrói o estado para a IA
        const state = [];
        for (let r = 0; r < this.boardSize; r++) {
            const rowState = [];
            for (let c = 0; c < this.boardSize; c++) {
                let cellValue = 0; // 0 = não visitado
                if (this.visited.has(`${r},${c}`)) {
                    cellValue = 1; // 1 = visitado
                }
                if (this.currentPosition.row === r && this.currentPosition.col === c) {
                    cellValue = 2; // 2 = posição atual
                }
                rowState.push(cellValue);
            }
            state.push(rowState);
        }

        // 2. Envia o estado para o servidor Flask
        try {
            this.aiHintBtn.disabled = true;
            this.aiHintBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Pensando...';

            const response = await fetch('http://localhost:5001/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ board: state })
            });

            if (!response.ok) {
                throw new Error(`Erro do servidor: ${response.statusText}`);
            }

            const data = await response.json();
            const { d_row, d_col } = data.move;

            // 3. Destaca a célula sugerida
            const suggestedRow = this.currentPosition.row + d_row;
            const suggestedCol = this.currentPosition.col + d_col;

            if (this.isValidPosition(suggestedRow, suggestedCol)) {
                this.board[suggestedRow][suggestedCol].classList.add('ai-hint');
            }

        } catch (error) {
            console.error("Erro ao obter dica da IA:", error);
            alert("Não foi possível conectar ao servidor da IA. Verifique se o servidor Flask (app.py) está em execução.");
        } finally {
            this.aiHintBtn.disabled = false;
            this.aiHintBtn.innerHTML = '<i class="fas fa-robot"></i> Dica da IA';
        }
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new CavaloSolitario();
}); 