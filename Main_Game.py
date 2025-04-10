import math
from typing import List, Tuple

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, board: 'HexBoard') -> Tuple[int, int]:
        raise NotImplementedError("¡Implementa este método!")

class HexBoard:
    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]

    def clone(self) -> 'HexBoard':
        cloned = HexBoard(self.size)
        cloned.board = [row.copy() for row in self.board]
        return cloned

    def get_possible_moves(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        if self.board[row][col] == 0:
            self.board[row][col] = player_id
            return True
        return False

    def check_connection(self, player_id: int) -> bool:
        # Implementación simplificada (requiere BFS/DFS completo)
        visited = set()
        queue = []
        if player_id == 1:
            for j in range(self.size):
                if self.board[0][j] == 1:
                    queue.append((0, j))
        else:
            for i in range(self.size):
                if self.board[i][0] == 2:
                    queue.append((i, 0))
        
        while queue:
            i, j = queue.pop(0)
            if (player_id == 1 and i == self.size - 1) or (player_id == 2 and j == self.size - 1):
                return True
            for ni, nj in self.get_neighbors(i, j):
                if self.board[ni][nj] == player_id and (ni, nj) not in visited:
                    visited.add((ni, nj))
                    queue.append((ni, nj))
        return False

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        neighbors = []
        size = self.size
        if row % 2 == 0:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]
        else:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1)]
        
        for di, dj in dirs:
            ni, nj = row + di, col + dj
            if 0 <= ni < size and 0 <= nj < size:
                neighbors.append((ni, nj))
        return neighbors

class SmartPlayer(Player):
    def __init__(self, player_id: int, max_depth: int = 3):
        super().__init__(player_id)
        self.max_depth = max_depth

    def play(self, board: HexBoard) -> Tuple[int, int]:
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        for move in board.get_possible_moves():
            cloned = board.clone()
            cloned.place_piece(move[0], move[1], self.player_id)
            value = self.minimax(cloned, self.max_depth, alpha, beta, False)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
        return best_move

    def minimax(self, board: HexBoard, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        if depth == 0 or board.check_connection(self.player_id) or board.check_connection(3 - self.player_id):
            return self.heuristic(board)
        
        if maximizing:
            value = -math.inf
            for move in board.get_possible_moves():
                cloned = board.clone()
                cloned.place_piece(move[0], move[1], self.player_id)
                value = max(value, self.minimax(cloned, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = math.inf
            for move in board.get_possible_moves():
                cloned = board.clone()
                cloned.place_piece(move[0], move[1], 3 - self.player_id)
                value = min(value, self.minimax(cloned, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def heuristic(self, board: HexBoard) -> float:
        player_score = self.evaluate_connectivity(board, self.player_id)
        opponent_score = self.evaluate_connectivity(board, 3 - self.player_id)
        centrality = self.evaluate_centrality(board)
        return player_score - 2 * opponent_score + 0.5 * centrality

    def evaluate_connectivity(self, board: HexBoard, player_id: int) -> int:
        size = board.size
        distance = 0
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == player_id:
                    if player_id == 1:
                        distance += (size - 1 - i)  # Distancia al lado opuesto
                    else:
                        distance += (size - 1 - j)
        return distance

    def evaluate_centrality(self, board: HexBoard) -> int:
        center = (board.size // 2, board.size // 2)
        score = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == self.player_id:
                    score += board.size - abs(i - center[0]) - abs(j - center[1])
        return score