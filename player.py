from HexBoard import HexBoard
import math
from typing import List, Tuple,Dict, Optional
from heapq import heappush, heappop
# Definición de la clase base Player
# Esta clase representa un jugador en el juego Hex.
class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # 1 (rojo) or 2 (azul)
        self.player_tokens= set()

    def play(self, board: "HexBoard") -> tuple:
        raise NotImplementedError("Implement this method!")
    
# Este es el que tenemos que hacer nosotros
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
            clone=board.clone()
            clone.place_piece(move[0], move[1], self.player_id)
            value = self.minimax(clone, self.max_depth, alpha, beta, False)
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
                clone= board.clone()
                clone.place_piece(move[0], move[1], self.player_id)
                value = max(value, self.minimax(clone, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
                
            return value
        else:
            value = math.inf
            for move in board.get_possible_moves():
                clone=board.clone()
                clone.place_piece(move[0], move[1], 3 - self.player_id)
                value = min(value, self.minimax(clone, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    
                    break
                
            return value

    def heuristic(self, board: HexBoard) -> float:
        #player_score = self.evaluate_connectivity(board, self.player_id)
        #opponent_score = self.evaluate_connectivity(board, 3 - self.player_id)

        # Costo del camino mínimo para el jugador actual
        player_path_cost = self.a_star_path_cost(board, self.player_id)
        # Costo del camino mínimo para el oponente (invertido)
        opponent_path_cost = self.a_star_path_cost(board, 3 - self.player_id)
        centrality = self.evaluate_centrality(board)
        # Heurística: Minimizar costo propio y maximizar costo del oponente
        return (-player_path_cost * 2) + (opponent_path_cost) + (0.3 * centrality)

    
    def a_star_path_cost(self, board: HexBoard, player_id: int) -> float:
        size = board.size
        start_nodes = []
        goal = {}
        # Definir puntos de inicio y objetivo según el jugador
        for i in self.player_tokens:                
            start_nodes.append(i)
        if player_id == 1:  # Conectar izquierda (col 0) a derecha (col size-1)
            goal= {'1':lambda x, y: y == size - 1,'2':"right"}
        else:  # Jugador 2: Conectar arriba (fila 0) a abajo (fila size-1)
            goal= {'1':lambda x, y: x == size - 1,'2':"Down"}

        if not start_nodes:
            return float('inf')  # Si no hay inicio, costo infinito

        min_cost = float('inf')
        # Calcular camino mínimo desde cada punto de inicio posible
        for start in start_nodes:
            cost = self.a_star(board, start, goal, player_id)
            if cost < min_cost:
                min_cost = cost
        return min_cost

    def a_star(self, board: HexBoard, start: Tuple[int, int], goal, player_id: int) -> float:
        open_heap = []
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        heappush(open_heap, (0, start))

        while open_heap:
            current = heappop(open_heap)[1]
            current_x, current_y = current

            if goal[1](current_x, current_y):
                return g_score[current]

            for neighbor in self.get_neighbors(board,current_x, current_y):
                nx, ny = neighbor
                # Costo de movimiento: 1 para vacío, 100 para celdas del oponente, 0 para propias
                cell_cost = 1 if board.board[nx][ny] == 0 else (100 if board.board[nx][ny] != player_id else 0)
                tentative_g = g_score.get(current, float('inf')) + cell_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.h(board.size,nx, ny, goal)
                    heappush(open_heap, (f_score, neighbor))

        return float('inf')  # No hay camino

    def h(self,board_size, x: int, y: int, goals) -> float:
        # Heurística admisible: distancia Manhattan hacia el objetivo más cercano
        size = board_size-1
        if goals[2] == "derecha":
            return size - y  # Distancia a la columna derecha
        else:
            return size - x  # Distancia a la fila inferior

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

    def get_neighbors(self,board:HexBoard,row: int, col: int) -> List[Tuple[int, int]]:
        neighbors = []
        size = board.size
        if row % 2 == 0:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]
        else:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1)]
        
        for di, dj in dirs:
            ni, nj = row + di, col + dj
            if 0 <= ni < size and 0 <= nj < size and board.board[ni][nj]==0:
                neighbors.append((ni, nj))
                board.board[ni][nj]= self.player_id
        return neighbors