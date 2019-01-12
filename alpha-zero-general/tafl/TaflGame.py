import copy
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

    def __init__(self, size):
        if size != 11 and size != 9 and size != 7:
            raise ValueError
        self.size = size

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
        return self.size*self.size*self.size*2

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """

        action = self.action_conversion__index_to_explicit(action)

        board = copy.deepcopy(board)
        board.do_action(action, player)
        next_player = -1 if player == 1 else 1
        return board, next_player

    def action_conversion__index_to_explicit(self, action):
        from_x, from_y, to, movement_type = np.unravel_index(action, (self.size, self.size, self.size, 2))
        if movement_type == MovementType.horizontal:
            action = ((from_x + 1, from_y + 1), (to + 1, from_y + 1))  # all coordinates + 1 because of the border
        else:
            action = ((from_x + 1, from_y + 1), (from_x + 1, to + 1))  # all coordinates + 1 because of the border
        return action

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
        array = np.zeros((self.size, self.size, self.size, 2))  # see comment in getActionSize()
                                            # use 0 for horizontal movement indexing, 1 for vertical movement indexing
        for explicit in board.get_valid_actions(player):
            index=self.action_conversion__explicit_to_indices( explicit)
            array[index]=1

        assert array[0, 0, 0, 0] == 0
        return array.ravel()

    def action_conversion__explicit_to_indices(self, explicit):
        (x_from,y_from),(x_to,y_to)=explicit
        movement_type = MovementType.horizontal if y_from == y_to else MovementType.vertical
        to = x_to if movement_type == MovementType.horizontal else y_to
        return (x_from - 1, y_from - 1, to - 1, movement_type)   # all coordinates -1 because of the border

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

    def getCanonicalForm(self, board, player):
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
        return copy.deepcopy(board)

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # TODO implement
        actions_and_probs=[(index,prob)for index,prob in enumerate(pi) if prob !=0]


        symmetries=[]

        #original
        symmetries.append((board.board[1:self.size+1, 1:self.size+1], pi))

        if False:
            # vertical flip
            temp_board=np.copy(board.board[1:self.size+1, 1:self.size+1])
            np.flip(temp_board,0)
            temp_pi=np.zeros((self.size,self.size,self.size,2))
            for index, prob in actions_and_probs:
                ((x_from,y_from),(x_to,y_to))=self.action_conversion__index_to_explicit(index)
                explicit=(self.size+1 - x_from, y_from),(self.size + 1- x_to, y_to)
                temp_pi[self.action_conversion__explicit_to_indices(explicit)]=prob
            temp_pi.ravel()
            symmetries.append((temp_board,temp_pi))

            # horizontal and vertical flip

            np.flip(temp_board, 1)
            temp_pi = np.zeros((self.size, self.size, self.size, 2))
            for index, prob in actions_and_probs:
                ((x_from, y_from), (x_to, y_to)) = self.action_conversion__index_to_explicit(index)
                explicit = (self.size + 1 - x_from, self.size + 1 - y_from), (self.size + 1 - x_to, self.size + 1 - y_to)
                temp_pi[self.action_conversion__explicit_to_indices(explicit)]=prob
            temp_pi.ravel()
            symmetries.append((temp_board, temp_pi))

            # horizontal flip

            np.flip(temp_board, 0)
            temp_pi = np.zeros((self.size, self.size, self.size, 2))
            for index, prob in actions_and_probs:
                ((x_from, y_from), (x_to, y_to)) = self.action_conversion__index_to_explicit(index)
                explicit = (x_from, self.size + 1 - y_from), (x_to, self.size + 1 - y_to)
                temp_pi[self.action_conversion__explicit_to_indices(explicit)]=prob
            temp_pi.ravel()
            symmetries.append((temp_board, temp_pi))



        return symmetries

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(board)
