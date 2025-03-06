import sys
from copy import deepcopy
from search import (
    Problem,
    Node,
    astar_search,
    depth_first_tree_search,
    breadth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    
)


class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id


class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    def __init__(self, matriz):
        self.grid = matriz
        self.actions_matrix = [[(0, []) for _ in range(len(matriz[0]))] for _ in range(len(matriz))]




    def get_value(self, row: int, col: int):
        return self.grid[row][col]

    def adjacent_vertical_values(self, row: int, col: int):
        above = self.grid[row-1][col] if row > 0 else None
        below = self.grid[row+1][col] if row < len(self.grid)-1 else None
        return above, below

    def adjacent_horizontal_values(self, row: int, col: int):
        left = self.grid[row][col-1] if col > 0 else None
        right = self.grid[row][col+1] if col < len(self.grid[row])-1 else None
        return left, right

    def print_matriz(self):
        for row in self.grid:
            print("\t".join(row))



    @staticmethod
    def parse_instance():
        try:
            linhas = sys.stdin.readlines()
        except:
            exit()
        matriz = []
        for linha in linhas:
            linha = linha.strip().split("\t")
            matriz.append(linha)

        return Board(matriz)


class PipeMania(Problem):
    
    def __init__(self, board: Board):
        self.initial = PipeManiaState(board)
        self.calculate_actions(self.initial)


    def verify1(self, state: PipeManiaState, i: int, j: int, piece: str):
        rows = len(state.board.grid)
        cols = len(state.board.grid[0])

        piece_actions = []

        if state.board.get_value(i, j) == piece:
            if i == 0 and j == 0:
                for new_piece in ['FB', 'FD', 'VB']:
                    if new_piece[0] == piece[0]:
                        piece_actions.append((i, j, new_piece))
            elif i == 0 and j == cols - 1:
                for new_piece in ['FB', 'FE', 'VE']:
                    if new_piece[0] == piece[0]:
                        piece_actions.append((i, j, new_piece))
            elif i == rows - 1 and j == 0:
                for new_piece in ['FC', 'FD', 'VD']:
                    if new_piece[0] == piece[0]:
                        piece_actions.append((i, j, new_piece))
            elif i == rows - 1 and j == cols - 1:
                for new_piece in ['FC', 'FE', 'VC']:
                    if new_piece[0] == piece[0]:
                        piece_actions.append((i, j, new_piece))
            else:
                if i == 0:
                    for new_piece in ['FB', 'FD', 'FE', 'BB', 'VB', 'VE', 'LH']:
                        if new_piece[0] == piece[0]:
                            piece_actions.append((i, j, new_piece))
                elif i == rows - 1:
                    for new_piece in ['FC', 'FE', 'FD', 'BC', 'VC', 'VD', 'LH']:
                        if new_piece[0] == piece[0]:
                            piece_actions.append((i, j, new_piece))
                elif j == 0:
                    for new_piece in ['FC', 'FB', 'FD', 'BD', 'VB', 'VD', 'LV']:
                        if new_piece[0] == piece[0]:
                            piece_actions.append((i, j, new_piece))
                elif j == cols - 1:
                    for new_piece in ['FB', 'FC', 'FE', 'BE', 'VC', 'VE', 'LV']:
                        if new_piece[0] == piece[0]:
                            piece_actions.append((i, j, new_piece))
                else:
                    for new_piece in ['FB', 'FC', 'FD', 'FE', 'BB', 'BC', 'BD', 'BE', 'VB', 'VC', 'VD', 'VE', 'LH', 'LV']:
                        if new_piece[0] == piece[0]:
                            piece_actions.append((i, j, new_piece))
                            
                         
        return piece_actions

    
    
    def case_zero_not_found(self, state: PipeManiaState, rows, cols):
        piece_actions = []
        one_found = False
        for i in range(rows):
            for j in range(cols):
                if state.board.actions_matrix[i][j][0] == 1:
                    one_found = True
                    temp_action = state.board.actions_matrix[i][j][1][0]
                    piece_actions.append(temp_action)
                    state.board.actions_matrix[i][j] = (0, [])
                    
        if one_found == False:
            return False
        return piece_actions
    

    
    def case_search_negative(self, state: PipeManiaState, rows, cols):
        temp_actions = []
        possible_actions_for_best_piece = []
        for i in range(rows):
            for j in range(cols):
                if state.board.actions_matrix[i][j][0] > 1:
                    #adiciona a primeira ação possivel de uma peça com várias ações à lista de ações
                    temp_actions.append(state.board.actions_matrix[i][j][1][0])
        if temp_actions:
            best_option = self.chose_best_option(state, temp_actions)
            for action in state.board.actions_matrix[best_option[0]][best_option[1]][1]:
                possible_actions_for_best_piece.append([action])

            state.board.actions_matrix[best_option[0]][best_option[1]][1] = (0, [])
        
        return possible_actions_for_best_piece
    
    
    def chose_best_option(self, state: PipeManiaState, actions):

        best_option_changed = False
        best_option = None
        for action in actions:
            i, j = action[0], action[1]
                    
            if i == 0 and j == 0 or i == 0 and j == len(state.board.grid[0]) - 1 or i == len(state.board.grid) - 1 and j == 0 or i == len(state.board.grid) - 1 and j == len(state.board.grid[0]) - 1:
                #se a ação for melhor que a melhor ação encontrada até ao momento e de coordenadas diferentes, atualiza a melhor ação
                if best_option == None:
                    best_option = [0,action]
                    best_option_changed = True
                elif (best_option[0] > 0) and (best_option[1][0] != i or best_option[1][1] != j):
                    best_option = [0,action]
                    best_option_changed = True
                    
            elif i == 0 or i == len(state.board.grid) - 1 or j == 0 or j == len(state.board.grid[0]) - 1:
                if best_option == None:
                    best_option = [1,action]
                    best_option_changed = True
                elif best_option[0] > 1:
                    best_option = [1,action]
                    best_option_changed = True
                    
            elif state.board.actions_matrix[i][j][0] == 2:
                if best_option == None:
                    best_option = [2,action]
                    best_option_changed = True
                elif best_option[0] > 2:
                    best_option = [2,action]
                    best_option_changed = True
                    
            elif state.board.actions_matrix[i][j][0] == 3:
                if best_option == None:
                    best_option = [3,action]
                    best_option_changed = True
                if best_option[0] > 3:
                    best_option = [3,action]
                    best_option_changed = True
        
        if best_option_changed:
            return best_option[1]
        return actions[0]
        

            

    def actions(self, state: PipeManiaState):
        rows = len(state.board.grid)
        cols = len(state.board.grid[0])
        piece_actions = []
        one_found = False
        zero_found = False
        for i in range(rows):
            for j in range(cols):
                if state.board.actions_matrix[i][j][0] == 0:
                    zero_found = True
                    test_verify = self.verify_neighbours(i, j, state, piece_actions)
                    piece_actions = test_verify[2]
                    if test_verify[0]:
                        #para cada ação em teste_verify[1] junta essa ação à lista de ações piece_actions
                        for action in test_verify[1]:
                            if state.board.actions_matrix[action[0]][action[1]][0] == 0 or state.board.actions_matrix[action[0]][action[1]][0] == 1:
                                if action not in piece_actions:
                                    piece_actions.append(action)
                    state.board.actions_matrix[i][j] = (-1, [])
                elif state.board.actions_matrix[i][j][0] == 1:
                    one_found = True
                    temp_action = state.board.actions_matrix[i][j][1][0]
                    if temp_action not in piece_actions:
                        piece_actions.append(temp_action)
                    state.board.actions_matrix[i][j] = (0, [])
        if zero_found and piece_actions:
            piece_actions = sorted(piece_actions)
            return [piece_actions]



        if not (one_found and not zero_found) or not piece_actions:
            piece_actions = self.case_search_negative(state, rows, cols)
            piece_actions = sorted(piece_actions)
            
            return piece_actions
        
        return [piece_actions]



    def verify_neighbours(self, i, j, state: PipeManiaState, actions_previously_found):
        board = state.board
        rows, cols = len(board.grid), len(board.grid[0])
        current_piece = board.grid[i][j]
        actions_found = []

        directions = [
            (-1, 0, 'above', 'below'),  # direção (di, dj, posição atual, posição do vizinho)
            (1, 0, 'below', 'above'),
            (0, -1, 'left', 'right'),
            (0, 1, 'right', 'left')
        ]


        for di, dj, pos, neighbour_pos in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if (board.actions_matrix[ni][nj][0] != -1) and (board.actions_matrix[ni][nj][0] != 0):
                    new_actions = []
                    for action in list(board.actions_matrix[ni][nj][1]):
                        if self.is_piece_connected(current_piece, i, j, action[2], pos, board):
                            new_actions.append(action)
                        else:
                            if pos == 'right' and self.connects_to_left(action[2]):
                                if action in actions_previously_found:
                                    actions_previously_found.remove(action)
                                board.actions_matrix[ni][nj][1].remove(action)
                                board.actions_matrix[ni][nj][0] -=1
                            if pos == 'left' and self.connects_to_right(action[2]):
                                if action in actions_previously_found:
                                    actions_previously_found.remove(action)
                                board.actions_matrix[ni][nj][1].remove(action)
                                board.actions_matrix[ni][nj][0] -=1
                            if pos == 'above' and self.connects_to_bottom(action[2]):
                                if action in actions_previously_found:
                                    actions_previously_found.remove(action)
                                board.actions_matrix[ni][nj][1].remove(action)
                                board.actions_matrix[ni][nj][0] -=1
                            if pos == 'below' and self.connects_to_top(action[2]):
                                if action in actions_previously_found:
                                    actions_previously_found.remove(action)
                                board.actions_matrix[ni][nj][1].remove(action)
                                board.actions_matrix[ni][nj][0] -=1
                                    
                            if not self.connects_to_left(current_piece) and pos == 'left':
                                if self.connects_to_right(action[2]):
                                    if action in board.actions_matrix[ni][nj][1]:
                                        board.actions_matrix[ni][nj][1].remove(action)
                                        board.actions_matrix[ni][nj][0] -=1

                            if not self.connects_to_right(current_piece) and pos == 'right':
                                if self.connects_to_left(action[2]):
                                    if action in board.actions_matrix[ni][nj][1]:
                                        board.actions_matrix[ni][nj][1].remove(action)
                                        board.actions_matrix[ni][nj][0] -=1
                            if not self.connects_to_bottom(current_piece) and pos == 'below':
                                if self.connects_to_top(action[2]):
                                    if action in board.actions_matrix[ni][nj][1]:
                                        board.actions_matrix[ni][nj][1].remove(action)
                                        board.actions_matrix[ni][nj][0] -=1
                            if not self.connects_to_top(current_piece) and pos == 'above':
                                if self.connects_to_bottom(action[2]):
                                    if action in board.actions_matrix[ni][nj][1]:
                                        board.actions_matrix[ni][nj][1].remove(action)
                                        board.actions_matrix[ni][nj][0] -=1
                            
                                
                    if new_actions:
                        board.actions_matrix[ni][nj] = [len(new_actions), new_actions]
                        actions_found.extend(new_actions)
      

        actions_found = list(set(actions_found))

        return bool(actions_found), actions_found, actions_previously_found


    #preenche a matriz secundária com o numero de ações possíveis de cada peça
    def calculate_actions(self, state: PipeManiaState):
        rows = len(state.board.grid)
        cols = len(state.board.grid[0])

        for i in range(rows):
            for j in range(cols):
                piece_actions = self.verify1(state, i, j, state.board.grid[i][j])
                if len(piece_actions) > 0:
                    state.board.actions_matrix[i][j] = [len(piece_actions), piece_actions]
        
        for i in range(rows):
            for j in range(cols):
                piece = state.board.grid[i][j]
                if piece.startswith('F'):
                    above, below = state.board.adjacent_vertical_values(i,j)   
                    left, right = state.board.adjacent_horizontal_values(i,j)  
                    actions_for_piece = state.board.actions_matrix[i][j]
                    if above != None and above.startswith('F') and (i, j, 'FC') in actions_for_piece[1]:
                        state.board.actions_matrix[i][j][1].remove((i,j,'FC'))
                        state.board.actions_matrix[i][j][0] -=1
                    if below != None and below.startswith('F') and (i, j, 'FB') in actions_for_piece[1]:
                        state.board.actions_matrix[i][j][1].remove((i,j,'FB'))
                        state.board.actions_matrix[i][j][0] -=1
                    if right != None and right.startswith('F') and (i, j, 'FD') in actions_for_piece[1]:
                        state.board.actions_matrix[i][j][1].remove((i,j,'FD'))
                        state.board.actions_matrix[i][j][0] -=1
                    if left != None and left.startswith('F') and (i, j, 'FE') in actions_for_piece[1]:
                        state.board.actions_matrix[i][j][1].remove((i,j,'FE'))       
                        state.board.actions_matrix[i][j][0] -=1
    
    def result(self, state: PipeManiaState, actions):
        new_board = deepcopy(state.board)
        for action in actions:
            i, j, new_piece = action
            new_board.grid[i][j] = new_piece
            new_board.actions_matrix[i][j] = (0, [])
        
        new_state = PipeManiaState(new_board)
      
        return new_state


    def check_auxiliar_connection(self, neighbour_piece, neighbour_piece_row, neighbour_piece_col, board, pieces_that_connect_from_current_to_top, pieces_that_connect_from_current_to_bottom, pieces_that_connect_from_current_to_left, pieces_that_connect_from_current_to_right, possible_pieces_list):
        possible_actions = possible_pieces_list.copy()
        
        max_row = len(board.grid) - 1
        max_col = len(board.grid[0]) - 1
        
        if neighbour_piece == 'BC':
            if neighbour_piece_col - 1 >= 0 and (board.actions_matrix[neighbour_piece_row][neighbour_piece_col-1][0] == -1) and (board.grid[neighbour_piece_row][neighbour_piece_col-1] not in pieces_that_connect_from_current_to_left):
                if 'BC' in possible_actions:
                    possible_actions.remove('BC')
            if neighbour_piece_col + 1 <= max_col and (board.actions_matrix[neighbour_piece_row][neighbour_piece_col+1][0] == -1) and (board.grid[neighbour_piece_row][neighbour_piece_col+1] not in pieces_that_connect_from_current_to_right):
                if 'BC' in possible_actions:
                    possible_actions.remove('BC')
            if neighbour_piece_row - 1 >= 0 and (board.actions_matrix[neighbour_piece_row-1][neighbour_piece_col][0] == -1) and (board.grid[neighbour_piece_row-1][neighbour_piece_col] not in pieces_that_connect_from_current_to_top):
                if 'BC' in possible_actions:
                    possible_actions.remove('BC')
        elif neighbour_piece == 'BB':
            if neighbour_piece_col - 1 >= 0 and (board.actions_matrix[neighbour_piece_row][neighbour_piece_col-1][0] == -1) and (board.grid[neighbour_piece_row][neighbour_piece_col-1] not in pieces_that_connect_from_current_to_left):
                if 'BB' in possible_actions:
                    possible_actions.remove('BB')
            if neighbour_piece_col + 1 <= max_col and (board.actions_matrix[neighbour_piece_row][neighbour_piece_col+1][0] == -1) and (board.grid[neighbour_piece_row][neighbour_piece_col+1] not in pieces_that_connect_from_current_to_right):
                if 'BB' in possible_actions:
                    possible_actions.remove('BB')
            if neighbour_piece_row + 1 <= max_row and (board.actions_matrix[neighbour_piece_row+1][neighbour_piece_col][0] == -1) and (board.grid[neighbour_piece_row+1][neighbour_piece_col] not in pieces_that_connect_from_current_to_bottom):
                if 'BB' in possible_actions:
                    possible_actions.remove('BB')
        elif neighbour_piece == 'BE':
            if neighbour_piece_col -1 <= max_col and (board.actions_matrix[neighbour_piece_row][neighbour_piece_col-1][0] == -1) and (board.grid[neighbour_piece_row][neighbour_piece_col-1] not in pieces_that_connect_from_current_to_left):
                if 'BE' in possible_actions:
                    possible_actions.remove('BE')
            if neighbour_piece_row - 1 >= 0 and (board.actions_matrix[neighbour_piece_row-1][neighbour_piece_col][0] == -1) and (board.grid[neighbour_piece_row-1][neighbour_piece_col] not in pieces_that_connect_from_current_to_top):
                if 'BE' in possible_actions:
                    possible_actions.remove('BE')
            if neighbour_piece_row + 1 <= max_row and (board.actions_matrix[neighbour_piece_row+1][neighbour_piece_col][0] == -1) and (board.grid[neighbour_piece_row+1][neighbour_piece_col] not in pieces_that_connect_from_current_to_bottom):
                if 'BE' in possible_actions:
                    possible_actions.remove('BE')
        elif neighbour_piece == 'BD':
            if neighbour_piece_col + 1 >= 0 and (board.actions_matrix[neighbour_piece_row][neighbour_piece_col+1][0] == -1) and (board.grid[neighbour_piece_row][neighbour_piece_col+1] not in pieces_that_connect_from_current_to_right):
                if 'BD' in possible_actions:
                    possible_actions.remove('BD')
            if neighbour_piece_row - 1 >= 0 and (board.actions_matrix[neighbour_piece_row-1][neighbour_piece_col][0] == -1) and (board.grid[neighbour_piece_row-1][neighbour_piece_col] not in pieces_that_connect_from_current_to_top):
                if 'BD' in possible_actions:
                    possible_actions.remove('BD')
            if neighbour_piece_row + 1 <= max_row and (board.actions_matrix[neighbour_piece_row+1][neighbour_piece_col][0] == -1) and (board.grid[neighbour_piece_row+1][neighbour_piece_col] not in pieces_that_connect_from_current_to_bottom):
                if 'BD' in possible_actions:
                    possible_actions.remove('BD')

        return possible_actions


    def connects_to_top (self, piece):
        pieces_that_connect_from_current_to_bottom = ['FC', 'BC', 'BE', 'BD', 'VC', 'VD', 'LV']
        if piece in pieces_that_connect_from_current_to_bottom:
            return True
        return False

    def connects_to_bottom (self, piece):
        pieces_that_connect_from_current_to_top = ['FB', 'BB', 'BE', 'BD', 'VB', 'VE', 'LV']
        if piece in pieces_that_connect_from_current_to_top:
            return True
        return False
    
    def connects_to_left (self, piece):
        pieces_that_connect_from_current_to_right = ['FE', 'BB', 'BC', 'BE', 'VC', 'VE', 'LH']
        if piece in pieces_that_connect_from_current_to_right:
            return True
        return False
    
    def connects_to_right (self,piece):
        pieces_that_connect_from_current_to_left = ['FD', 'BC', 'BB', 'BD', 'VB', 'VD', 'LH']
        if piece in pieces_that_connect_from_current_to_left:
            return True
        return False




    def is_piece_connected(self, current_piece, current_piece_row, current_piece_col, neighbour_piece, direction, board):

        pieces_that_connect_from_current_to_top = ['FB', 'BB', 'BE', 'BD', 'VB', 'VE', 'LV']
        pieces_that_connect_from_current_to_bottom = ['FC', 'BC', 'BE', 'BD', 'VC', 'VD', 'LV']
        pieces_that_connect_from_current_to_left = ['FD', 'BC', 'BB', 'BD', 'VB', 'VD', 'LH']
        pieces_that_connect_from_current_to_right = ['FE', 'BB', 'BC', 'BE', 'VC', 'VE', 'LH']
        possible = []
        
        if self.connects_to_top(current_piece) and direction == 'above':
            possible = self.check_auxiliar_connection(neighbour_piece, current_piece_row-1, current_piece_col, board, pieces_that_connect_from_current_to_top, pieces_that_connect_from_current_to_bottom, pieces_that_connect_from_current_to_left, pieces_that_connect_from_current_to_right, pieces_that_connect_from_current_to_top)
            if current_piece == 'FC' and 'FB' in possible:
                possible.remove('FB')
            return neighbour_piece in possible
        elif self.connects_to_bottom(current_piece) and direction == 'below':
            possible = self.check_auxiliar_connection(neighbour_piece, current_piece_row+1, current_piece_col, board, pieces_that_connect_from_current_to_top, pieces_that_connect_from_current_to_bottom, pieces_that_connect_from_current_to_left, pieces_that_connect_from_current_to_right, pieces_that_connect_from_current_to_bottom)
            if current_piece == 'FB' and 'FC' in possible:
                possible.remove('FC')
            return neighbour_piece in possible
            
        elif self.connects_to_left(current_piece) and direction == 'left':
            possible = self.check_auxiliar_connection(neighbour_piece, current_piece_row, current_piece_col-1, board, pieces_that_connect_from_current_to_top, pieces_that_connect_from_current_to_bottom, pieces_that_connect_from_current_to_left, pieces_that_connect_from_current_to_right, pieces_that_connect_from_current_to_left)
            if current_piece == 'FE' and 'FD' in possible:
                possible.remove('FD')
            return neighbour_piece in possible
        elif self.connects_to_right(current_piece) and direction == 'right':
            possible = self.check_auxiliar_connection(neighbour_piece, current_piece_row, current_piece_col+1, board, pieces_that_connect_from_current_to_top, pieces_that_connect_from_current_to_bottom, pieces_that_connect_from_current_to_left, pieces_that_connect_from_current_to_right, pieces_that_connect_from_current_to_right)
            if current_piece == 'FD' and 'FE' in possible:
                possible.remove('FE')
            return neighbour_piece in possible
        
        return False




    def pieces_fit_together(self, piece1, piece2, orientation):
        if orientation == 'vertical':
            return (piece1.endswith('B') and piece2.endswith('C')) or (piece1.endswith('C') and piece2.endswith('B'))
        elif orientation == 'horizontal':
            return (piece1.endswith('D') and piece2.endswith('E')) or (piece1.endswith('E') and piece2.endswith('D'))
        return False
    

    def goal_test(self, state: PipeManiaState):
        is_goal = self.is_goal(state)
        return is_goal


    def is_goal(self, state: PipeManiaState):
        visited = set()
        first_piece = (0, 0)

        # Realiza a DFS a partir da primeira peça
        self.dfs(state, first_piece[0], first_piece[1], visited)

        for i in range(len(state.board.grid)):
            for j in range(len(state.board.grid[i])):
                if (i, j) not in visited:
                    return False

        return True
    
    def dfs(self, state, i, j, visited):
        stack = [(i, j)]
        while stack:
            x, y = stack.pop()
            if (x, y) not in visited:
                visited.add((x, y))
                above, below = state.board.adjacent_vertical_values(x, y)
                left, right = state.board.adjacent_horizontal_values(x, y)
                
                # Adiciona os vizinhos conectados à pilha
                if above is not None and self.connects_to_bottom(above) and self.connects_to_top(state.board.grid[x][y]):
                    stack.append((x-1, y))
                if below is not None and self.connects_to_top(below) and self.connects_to_bottom(state.board.grid[x][y]):
                    stack.append((x+1, y))
                if left is not None and self.connects_to_right(left) and self.connects_to_left(state.board.grid[x][y]):
                    stack.append((x, y-1))
                if right is not None and self.connects_to_left(right) and self.connects_to_right(state.board.grid[x][y]):
                    stack.append((x, y+1))



    def h(self, node: Node):
        return self.number_of_unconnected_pieces(node.state)

    def number_of_unconnected_pieces(self, state: PipeManiaState):
        count = 0
        for i in range(len(state.board.grid)):
            for j in range(len(state.board.grid[i])):
                if state.board.actions_matrix[i][j][0] != -1 and state.board.actions_matrix[i][j][0] != 0:
                    count += 1
        return count
    

if __name__ == "__main__":
    board = Board.parse_instance()
    problem = PipeMania(board)

    solution_node = astar_search(problem)

    solution_state = solution_node.state
    solution_state.board.print_matriz()
