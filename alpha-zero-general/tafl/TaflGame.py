import copy
import random
from enum import IntEnum

import numpy as np

from Game import Game
from tafl.TaflBoard import Outcome, Player, TaflBoard


class MovementType(IntEnum):
    horizontal = 0
    vertical = 1

class TaflGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self, size, prune):
        if size != 11 and size != 9 and size != 7:
            raise ValueError
        self.size = size
        self.prune = prune
        self.prune_prob = 0.1

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return TaflBoard(self.size)

    def getBoardSize(self):
        """
            Returns:
                (x,y): a tuple of board dimensions
        """
        return self.size, self.size

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # size*size to select the piece to move
        # size for horizontal movement, size for vertical movement, so size*2 to select the action to take
        return self.size*self.size*self.size*2+1

    def getNextState(self, board, player, action, copy_board=False):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """

        if action == self.getActionSize() - 1:
            board.outcome = Outcome.black if player == Player.white else Player.black
            # assert board.outcome != Outcome.ongoing, str(player) + " selected 'no action', but had still moves left\n" \
            #                                          + str(board) + "\n" + str(list(board.get_valid_actions(player)))
        else:
            explicit = action_conversion__index_to_explicit(action, self.size)
            assert action_conversion__explicit_to_index(explicit, self.size) == action
            if copy_board:
                board = copy.deepcopy(board)
            board.do_action(explicit, player)
        next_player = -1 if player == 1 else 1
        return board, next_player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        array = np.zeros(self.getActionSize())  # see comment in getActionSize()
                                            # use 0 for horizontal movement indexing, 1 for vertical movement indexing

        if self.prune and random.random() < self.prune_prob:
            non_losing_moves = []
            # white:
            # preferences:
            #   1. king to corner
            #   2. king next to corner or king to empty side
            #   3. go to (2,2) or symmetrical equivalents when king can't be captured and there is no piece on
            #       (2,1) or (1,2)
            #   4. force same board state for the third time
            # 	5. prevent king capture
            if player == Player.white:
                # 1., 2. and 3.
                winning_move = board.get_king_escape_move()
                if winning_move is None:
                    move_set = board.get_valid_actions(player)
                    # 4.
                    for action in move_set:
                        if board.would_next_board_be_third(action):
                            winning_move = action
                            break
                    # 5.
                    if winning_move is None:
                        for action in move_set:
                            if not board.would_next_board_lead_to_opponent_winning(action, Player.white):
                                non_losing_moves.append(action)
            # black:
            # preferences:
            #   1. capture king
            #   2. force same board state for the third time
            # 	3. prevent king to corner
            # 	4. prevent king next to corner and prevent king to empty side
            #   5. prevent king going to (2,2) or symmetrical equivalents when king can't be captured and there is no
            #       piece on  (2,1) or (1,2)
            else:
                move_set = board.get_valid_actions(player)
                # 1.
                winning_move = board.get_king_capture_move(move_set)
                if winning_move is None:
                    # 2.
                    for action in move_set:
                        if board.would_next_board_be_third(action):
                            winning_move = action
                            break
                    # 3., 4. and 5.
                    if winning_move is None:
                        for action in move_set:
                            if not board.would_next_board_lead_to_opponent_winning(action, Player.black):
                                non_losing_moves.append(action)

            # set winning move if it exists
            if winning_move is not None:
                index = action_conversion__explicit_to_index(winning_move, self.size)
                array[index] = 1
            else:
                # set non losing moves if they exist, but not a winning move
                for explicit in non_losing_moves:
                    index = action_conversion__explicit_to_index(explicit, self.size)
                    array[index] = 1
            # set any move if both don't exist
            if winning_move is None and non_losing_moves == []:
                if len(move_set) == 0:
                    index = self.getActionSize()-1
                else:
                    index = action_conversion__explicit_to_index(random.choice(move_set), self.size)
                array[index] = 1

        else:
            no_immediate_loss_possible = False
            move_set = board.get_valid_actions(player)
            for action in move_set:
                if board.would_next_board_be_third(action):
                    array = np.zeros(self.getActionSize())
                    index = action_conversion__explicit_to_index(action, self.size)
                    assert action_conversion__index_to_explicit(index, self.size) == action
                    array[index] = 1
                    return array
                elif not board.would_next_board_lead_to_third(action, player):
                    index = action_conversion__explicit_to_index(action, self.size)
                    assert action_conversion__index_to_explicit(index, self.size) == action
                    array[index] = 1
                    no_immediate_loss_possible = True
            # if all possible moves lead to a loss...
            if not no_immediate_loss_possible:
                # ... check if there are actually moves that can be made. If not, just choose an impossible move and the
                # board class will report the correct outcome. This is necessary because it is only in this method here
                # that it is checked whether there are actually moves left. Meaning that the Outcome is set in the board
                # class already, but the network is still asked to select a move. Therefore we need to give the program
                # at least one move to choose from.
                if len(move_set) == 0:
                    index = self.getActionSize()-1
                else:
                    index = action_conversion__explicit_to_index(random.choice(move_set), self.size)
                array[index] = 1
        return array

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        if board.outcome == Outcome.ongoing:
            return 0
        elif board.outcome == Outcome.draw:
            return 0.000001
        elif board.outcome == Outcome.black:
            return 1 if player == Player.black else -1
        else:
            return -1 if player == Player.black else 1

    def getCanonicalForm(self, board, player, copy_board=False):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # canonical form isn't really possible because of the asymmetric nature of tafl
        if copy_board:
            board = copy.deepcopy(board)
        return board

    def getSymmetries(self, board, pi, king_position):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        actions_and_probs = [(index, prob)for index, prob in enumerate(pi) if prob != 0]
        king_x, king_y = king_position

        symmetries = []
        # occurrences = np.zeros(self.size*self.size*self.size*2)
        # for index, prob in actions_and_probs:
        #    ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
        #     explicit = (self.size + 1 - x_from, y_from), (self.size + 1 - x_to, y_to)
        #     occurrences[index] = 1 if board.would_next_board_be_second_third(2, explicit) else 0

        #original
        temp_board = np.copy(board.board[1:self.size + 1, 1:self.size + 1])
        symmetries.append((temp_board, pi, (king_x, king_y)))

        # horizontal flip
        temp_board = np.flip(temp_board, 0)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (self.size + 1 - x_from, y_from), (self.size + 1 - x_to, y_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (self.size + 1 - king_x, king_y)))

        # horizontal and vertical flip
        temp_board = np.flip(temp_board, 1)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (self.size + 1 - x_from, self.size + 1 - y_from), (self.size + 1 - x_to, self.size + 1 - y_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (self.size + 1 - king_x, self.size + 1 - king_y)))

        # vertical flip
        temp_board = np.flip(temp_board, 0)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (x_from, self.size + 1 - y_from), (x_to, self.size + 1 - y_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (king_x, self.size + 1 - king_y)))

        # rotation
        temp_board = np.flip(temp_board, 1)
        temp_board = np.rot90(temp_board)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (self.size + 1 - y_from, x_from), (self.size + 1 - y_to, x_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (self.size + 1 - king_y, king_x)))

        # rotation and horizontal flip
        temp_board = np.flip(temp_board, 0)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (y_from, x_from), (y_to, x_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (king_y, king_x)))

        # rotation and horizontal and vertical flip
        temp_board = np.flip(temp_board, 1)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (y_from, self.size + 1 - x_from), (y_to, self.size + 1 - x_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (king_y, self.size + 1 - king_x)))

        # rotation and vertical flip
        temp_board = np.flip(temp_board, 0)
        temp_pi = np.zeros(self.size * self.size * self.size * 2 + 1)
        for index, prob in actions_and_probs:
            if index == self.size * self.size * self.size * 2:
                temp_pi[index] = prob
            else:
                ((x_from, y_from), (x_to, y_to)) = action_conversion__index_to_explicit(index, self.size)
                explicit = (self.size + 1 - y_from, self.size + 1 - x_from), (self.size + 1 - y_to, self.size + 1 - x_to)
                temp_pi[action_conversion__explicit_to_index(explicit, self.size)] = prob
        symmetries.append((temp_board, temp_pi, (self.size + 1 - king_y, king_x)))

        return symmetries

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.bytes()
        # return str(board)


def action_conversion__explicit_to_index(explicit, size):
    (x_from, y_from), (x_to, y_to) = explicit
    movement_type = MovementType.horizontal if y_from == y_to else MovementType.vertical
    to = x_to if movement_type == MovementType.horizontal else y_to
    result = (((x_from - 1) * size + y_from - 1) * size + to - 1) * 2 + movement_type   # all coordinates -1 because of the border
    assert 0 <= result < size*size*size * 2
    return result


def action_conversion__index_to_explicit(action, size):
    from_x, from_y, to, movement_type = np.unravel_index(action, (size, size, size, 2))
    if movement_type == MovementType.horizontal:
        action = ((from_x + 1, from_y + 1), (to + 1, from_y + 1))  # all coordinates + 1 because of the border
    else:
        action = ((from_x + 1, from_y + 1), (from_x + 1, to + 1))  # all coordinates + 1 because of the border
    return action
