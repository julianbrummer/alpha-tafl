import numpy as np
from enum import IntEnum

from tafl.RuleConfig import MAX_NUMBER_OF_TURNS, MAX_NUMBER_OF_TURNS_WITHOUT_CAPTURE


class Player(IntEnum):
    white = -1
    black = 1


class TileState(IntEnum):
    empty = 0   # neutral
    white = 1   # hostile to black
    black = 2   # hostile to white and king
    king = 4    # hostile to black
    throne = 8  # the empty throne is hostile to any piece
    corner = 16  # target tile for the king, hostile to any piece
    border = 32  # not a reachable tile


class Outcome(IntEnum):
    ongoing = 0
    white = -1
    black = 1
    draw = 2


class TaflBoard:

    def __init__(self, size):
        self.size = size
        # empty
        self.board = np.zeros((self.size + 2, self.size + 2), dtype=np.uint8)         # TileStates

        self.king_position = ((self.size + 1)/2, (self.size + 1)/2)

        # holds all board states and the frequency how often they occurred
        self.board_states_dict = {self.board.tobytes(): 1}

        # the outcome of the current match
        self.outcome = Outcome.ongoing

        # number of pieces of a color. white pieces includes the king
        self.white_pieces = 0
        self.black_pieces = 0

        # turn counts for draw condition
        self.turn_count = 0
        self.turns_without_capture_count = 0

        # Whether this board prints moves and captures to the console. This shouldn't
        # happen when the game tree is searched because during that time only
        # possibilities are considered, but no actual move is made. Turn this off for
        # all instances of this class that are a copy of the original class.
        self.print_to_console = False

        self.print_game_over_reason = False

        self.save_game = False

        # # for reverting actions
        # self.board_stack = []
        # self.king_position_stack = []
        # self.action_stack = []
        # self.turns_without_capture_count_stack = []

        # self.test_board()
        self.reset_board()

    def reset_board(self):
        # empty
        self.board = np.zeros((self.size + 2, self.size + 2), dtype=np.uint8)        # TileStates

        if self.size == 11:
            # black
            self.board[1, 4:9] = TileState.black
            self.board[2, 6] = TileState.black
            self.board[11, 4:9] = TileState.black
            self.board[10, 6] = TileState.black
            self.board[4:9, 1] = TileState.black
            self.board[6, 2] = TileState.black
            self.board[4:9, 11] = TileState.black
            self.board[6, 10] = TileState.black
            # white (fortress)
            self.board[5:8, 5:8] = TileState.white
            self.board[4, 6] = TileState.white
            self.board[6, 8] = TileState.white
            self.board[6, 4] = TileState.white
            self.board[8, 6] = TileState.white

            self.white_pieces = 13
            self.black_pieces = 24
        elif self.size == 7:
            # black
            self.board[1:3, 4] = TileState.black
            self.board[4, 1:3] = TileState.black
            self.board[4, 6:8] = TileState.black
            self.board[6:8, 4] = TileState.black
            # white
            self.board[3, 4] = TileState.white
            self.board[4, 3] = TileState.white
            self.board[4, 5] = TileState.white
            self.board[5, 4] = TileState.white

            self.white_pieces = 5
            self.black_pieces = 8
        elif self.size == 9:
            # black
            self.board[1, 4:7] = TileState.black
            self.board[2, 5] = TileState.black
            self.board[9, 4:7] = TileState.black
            self.board[8, 5] = TileState.black
            self.board[4:7, 1] = TileState.black
            self.board[5, 2] = TileState.black
            self.board[4:7, 9] = TileState.black
            self.board[5, 8] = TileState.black
            # white
            self.board[3:5, 5] = TileState.white
            self.board[5, 3:5] = TileState.white
            self.board[5, 6:8] = TileState.white
            self.board[6:8, 5] = TileState.white

            self.white_pieces = 9
            self.black_pieces = 16
        else:
            raise NotImplementedError

        # king
        self.king_position = ((self.size + 1)//2, (self.size + 1)//2)
        self.board[int(self.size + 1)//2, int(self.size + 1)//2] = TileState.king | TileState.throne
        # border
        self.board[0, :] = TileState.border
        self.board[self.size + 1, :] = TileState.border
        self.board[:, 0] = TileState.border
        self.board[:, self.size + 1] = TileState.border
        # corner
        self.board[1, 1] = TileState.corner
        self.board[1, self.size] = TileState.corner
        self.board[self.size, self.size] = TileState.corner
        self.board[self.size, 1] = TileState.corner

        self.board_states_dict = {self.board.tobytes(): 1}
        self.outcome = Outcome.ongoing
        self.turn_count = 0
        self.turns_without_capture_count = 0

    #  Checks whether "player" can do action "move".
    #  move = ((fromX,fromY),(toX,toY))
    def can_do_action(self, move, player):
        (position_from, position_to) = move
        player_tilestate = TileState.black if player == Player.black else TileState.white | TileState.king
        return self.board[position_from] & player_tilestate != 0 and move in self.get_valid_actions_for_piece(position_from)

    # returns all valid actions for a player as a list of actions
    def get_valid_actions(self, turn_player):
        valid_actions = []
        player_tilestate = TileState.black if turn_player == Player.black else TileState.white | TileState.king
        for position, tilestate in np.ndenumerate(self.board):
            if tilestate & player_tilestate != 0:
                valid_actions.extend(self.get_valid_actions_for_piece(position))
        if len(valid_actions) == 0:
            self.outcome = Outcome.white if turn_player == Player.black else Outcome.black
            if self.print_to_console or self.print_game_over_reason:
                print("It is " + str(turn_player) + "'s turn, but they can't make any moves. "
                      + str(Player.white if turn_player == Player.black else Player.black) + " wins!")
        return valid_actions

    # returns all valid actions for a piece at a given position as a list of actions
    def get_valid_actions_for_piece(self, position):
        x, y = position
        is_king = self.board[x, y] == TileState.king
        valid_actions = []
        # first direction
        for x_other in reversed(range(1, x)):
            if self.board[x_other, y] == TileState.empty or \
                    (is_king and self.board[x_other, y] & (TileState.corner | TileState.throne) != 0):
                valid_actions.append(((x, y), (x_other, y)))
            else:
                if self.board[x_other, y] != TileState.throne:
                    break
        # second direction
        for x_other in range(x + 1, self.size + 1):
            if self.board[x_other, y] == TileState.empty or \
                    (is_king and self.board[x_other, y] & (TileState.corner | TileState.throne) != 0):
                valid_actions.append(((x, y), (x_other, y)))
            else:
                if self.board[x_other, y] != TileState.throne:
                    break
        # third direction
        for y_other in reversed(range(1, y)):
            if self.board[x, y_other] == TileState.empty or \
                    (is_king and self.board[x, y_other] & (TileState.corner | TileState.throne) != 0):
                valid_actions.append(((x, y), (x, y_other)))
            else:
                if self.board[x, y_other] != TileState.throne:
                    break
        # forth direction
        for y_other in range(y + 1, self.size + 1):
            if self.board[x, y_other] == TileState.empty or \
                    (is_king and self.board[x, y_other] & (TileState.corner | TileState.throne) != 0):
                valid_actions.append(((x, y), (x, y_other)))
            else:
                if self.board[x, y_other] != TileState.throne:
                    break
        return valid_actions

    # executes "move" for the player "player" whose turn it is
    # except when the game is already over. In this case it does nothing
    def do_action(self, move, player):
        # return immediately if game over
        if self.outcome != Outcome.ongoing:
            return

        if self.save_game:
            print("File opened")
            url = __file__[:-28]
            game_filename = url + '/' + 'game.txt'
            file = open(game_filename,"a+")
            entry = ""
            entry += str(move) + " \n"

        (from_x, from_y), (to_x, to_y) = move
        if self.can_do_action(move, player):
            # increase turn counts
            # self.turn_count += 1
            # self.turns_without_capture_count_stack.append(self.turns_without_capture_count)
            # self.turns_without_capture_count += 1
            # self.action_stack.append((move, self.board[from_x, from_y]))
            # board_bytes = self.board.tobytes()
            # self.board_stack.append(board_bytes)
            # self.king_position_stack.append(self.king_position)

            if self.print_to_console:
                print(str(player) + " moves a piece from " + str((from_x, from_y)) + " to " + str((to_x, to_y)))
            # if king is moving: update king position and check if he reached a corner
            if self.board[from_x, from_y] & TileState.king != 0:
                self.king_position = (to_x, to_y)
                if self.board[self.king_position] == TileState.corner:
                    self.outcome = Outcome.white
                    if self.print_to_console or self.print_game_over_reason:
                        print("The king escapes to corner " + str((to_x, to_y)) + ". White wins!")
            # update the board itself and capture pieces if applicable

            # keep throne tile state if it is there and move the piece from the other tile here
            self.board[to_x, to_y] = (self.board[to_x, to_y] & TileState.throne) |\
                                     (self.board[from_x, from_y] & (TileState.white | TileState.black | TileState.king))
            # clear the old tile from pieces
            self.board[from_x, from_y] = self.board[from_x, from_y] & \
                                         ~(TileState.white | TileState.black | TileState.king)  # remove piece from tile
            captured_pieces = self.capture((to_x, to_y), player)

            # update the board_states_dictionary so that we know whether the present board has occurred for the 3rd time
            board_bytes = self.board.tobytes()
            if board_bytes in self.board_states_dict:
                self.board_states_dict[board_bytes] += 1
                if self.board_states_dict[board_bytes] == 3:
                    if player == Player.black:
                        self.outcome = Outcome.black
                        if self.print_to_console or self.print_game_over_reason:
                            print("White forced the same board state for third time. Black wins!")
                    else:
                        self.outcome = Outcome.white
                        if self.print_to_console or self.print_game_over_reason:
                            print("Black forced the same board state for third time. White wins!")
            else:
                self.board_states_dict[board_bytes] = 1


            # # check if draw conditions by turn count are met does not change if max_number_of_turns is  infty
            # if self.turn_count == MAX_NUMBER_OF_TURNS and self.outcome == Outcome.ongoing:
            #     self.outcome = Outcome.draw
            # if self.turns_without_capture_count == MAX_NUMBER_OF_TURNS_WITHOUT_CAPTURE \
            #         and self.outcome == Outcome.ongoing:
            #     self.outcome = Outcome.draw

            if self.save_game:
                if Outcome != Outcome.ongoing:
                    file.write(entry)
                    file.close()
            return captured_pieces
        else:
            raise Exception(str(Player(player)) + " tried to make move " + str(move) + ", but that move is not possible. "
                                                                               "Current board:\n" + self.__str__()
                            + "\npossible actions: " + str(self.get_valid_actions(player)))

    # captures all enemy pieces around the position "position_to" that the player "player" has just moved a piece to
    def capture(self, position_to, turn_player):
        x, y = position_to
        captured_pieces = []

        own_tile_state = TileState.black if turn_player == Player.black else TileState.white | TileState.king
        # TileState.white for Player.black, TileState.black for Player.white
        # this way is necessary because capturing the king works differently and is done further below
        opponent_pawn_tile_state = TileState.white if turn_player == Player.black else TileState.black

        # check capture right
        if self.board[x + 1, y] & opponent_pawn_tile_state != 0 \
                and self.board[x + 2, y] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            self.board[x + 1, y] = TileState.empty
            captured_pieces.append((x + 1, y))
            if self.print_to_console:
                print(str(turn_player) + " captures piece at " + str((x + 1, y)))

        # check capture left
        if self.board[x - 1, y] & opponent_pawn_tile_state != 0 \
                and self.board[x - 2, y] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            self.board[x - 1, y] = TileState.empty
            captured_pieces.append((x - 1, y))
            if self.print_to_console:
                print(str(turn_player) + " captures piece at " + str((x - 1, y)))

        # check capture bottom
        if self.board[x, y + 1] & opponent_pawn_tile_state != 0 \
                and self.board[x, y + 2] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            self.board[x, y + 1] = TileState.empty
            captured_pieces.append((x, y + 1))
            if self.print_to_console:
                print(str(turn_player) + " captures piece at " + str((x, y + 1)))

        # check capture top
        if self.board[x, y - 1] & opponent_pawn_tile_state != 0 \
                and self.board[x, y - 2] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            self.board[x, y - 1] = TileState.empty
            captured_pieces.append((x, y - 1))
            if self.print_to_console:
                print(str(turn_player) + " captures piece at " + str((x, y - 1)))

        # check capture king
        king_capture = False
        # check if this piece has moved next to the king at all (otherwise it would be impossible for the king to move
        # between two black pieces, as is done in the examples)
        if self.board[x + 1, y] & TileState.king != 0 \
                    and self.board[x + 2, y] & (own_tile_state | TileState.throne) != 0 \
                or self.board[x - 1, y] & TileState.king != 0 \
                    and self.board[x - 2, y] & (own_tile_state | TileState.throne) != 0 \
                or self.board[x, y + 1] & TileState.king != 0 \
                    and self.board[x, y + 2] & (own_tile_state | TileState.throne) != 0 \
                or self.board[x, y - 1] & TileState.king != 0 \
                    and self.board[x, y - 2] & (own_tile_state | TileState.throne) != 0:

            king_x, king_y = self.king_position
            # check: (king is on or next to throne and surrounded on all for sides)
            # or (between to black pieces in vertical direction)
            # or (between to black pieces in horizontal direction)
            if (self.board[king_x, king_y] | self.board[king_x + 1, king_y] | self.board[king_x - 1, king_y] |
                self.board[king_x, king_y + 1] | self.board[king_x, king_y - 1]) & TileState.throne != 0:
                if self.board[king_x + 1, king_y] & (TileState.black | TileState.throne) != 0 \
                        and self.board[king_x - 1, king_y] & (TileState.black | TileState.throne) != 0 \
                        and self.board[king_x, king_y + 1] & (TileState.black | TileState.throne) != 0 \
                        and self.board[king_x, king_y - 1] & (TileState.black | TileState.throne) != 0:
                    king_capture = True
            elif self.board[king_x + 1, king_y] & TileState.black != 0\
                    and self.board[king_x - 1, king_y] & TileState.black != 0\
                    or self.board[king_x, king_y + 1] & TileState.black != 0\
                    and self.board[king_x, king_y - 1] & TileState.black != 0:
                king_capture = True
            if king_capture:
                self.outcome = Outcome.black
                captured_pieces.append((king_x, king_y))
                if self.print_to_console or self.print_game_over_reason:
                    print("Black wins by capturing the king at " + str(self.king_position) + "!")

        # if len(captured_pieces) > 0:
        #     self.turns_without_capture_count = 0

        if turn_player == Player.white:
            self.black_pieces -= len(captured_pieces)
        else:
            self.white_pieces -= len(captured_pieces)

        return captured_pieces

    # checks whether the next move would lead to a board state where the opponent of the turn player could make a move
    # such that the resulting board state will be seen for the third time. This would lead to a win for the opponent
    def would_next_board_lead_to_third(self, move, turn_player):
        move_from, move_to = move
        x_to, y_to = move_to
        previous_from = self.board[move_from]
        previous_to = self.board[x_to, y_to]

        # check captures
        own_tile_state = TileState.black if previous_from & TileState.black != 0 else TileState.white | TileState.king
        # TileState.white for Player.black, TileState.black for Player.white
        # this way is necessary because capturing the king works differently and is done further below
        opponent_pawn_tile_state = own_tile_state ^ (TileState.black | TileState.white)

        # check capture right
        if self.board[x_to + 1, y_to] & opponent_pawn_tile_state != 0 \
                and self.board[x_to + 2, y_to] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            return False
        # check capture left
        if self.board[x_to - 1, y_to] & opponent_pawn_tile_state != 0 \
                and self.board[x_to - 2, y_to] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            return False
        # check capture bottom
        if self.board[x_to, y_to + 1] & opponent_pawn_tile_state != 0 \
                and self.board[x_to, y_to + 2] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            return False
        # check capture top
        if self.board[x_to, y_to - 1] & opponent_pawn_tile_state != 0 \
                and self.board[x_to, y_to - 2] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            return False

        # if nothing is captured and the current board state has been seen two times already,
        # then the next player can just revert the currently checked move and win
        board_bytes = self.board.tobytes()
        if board_bytes in self.board_states_dict and self.board_states_dict[board_bytes] == 2:
            return True

        # see regular move method for a short explanation
        self.board[move_to] = (self.board[move_to] & TileState.throne) | \
                              (self.board[move_from] & (TileState.white | TileState.black | TileState.king))
        self.board[move_from] = self.board[move_from] & \
                                ~(TileState.white | TileState.black | TileState.king)  # remove piece from tile
        result = False
        for action in self.get_valid_actions(-1 * turn_player):
            result = result or self.would_next_board_be_third(action)
        self.board[move_from] = previous_from
        self.board[move_to] = previous_to
        return result

    # checks whether the next move would lead to a board state that has occurred two times already, thus leading
    # to a win for the player exevuting that move
    def would_next_board_be_third(self, action):
        move_from, move_to = action
        x_to, y_to = move_to
        previous_from = self.board[move_from]
        previous_to = self.board[x_to, y_to]

        # check captures
        own_tile_state = TileState.black if previous_from & TileState.black != 0 else TileState.white | TileState.king
        # TileState.white for Player.black, TileState.black for Player.white
        # this way is necessary because capturing the king works differently and is done further below
        opponent_pawn_tile_state = own_tile_state ^ (TileState.black | TileState.white)

        # check capture right
        if self.board[x_to + 1, y_to] & opponent_pawn_tile_state != 0 \
                and self.board[x_to + 2, y_to] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            return False
        # check capture left
        if self.board[x_to - 1, y_to] & opponent_pawn_tile_state != 0 \
                and self.board[x_to - 2, y_to] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            return False
        # check capture bottom
        if self.board[x_to, y_to + 1] & opponent_pawn_tile_state != 0 \
                and self.board[x_to, y_to + 2] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            return False
        # check capture top
        if self.board[x_to, y_to - 1] & opponent_pawn_tile_state != 0 \
                and self.board[x_to, y_to - 2] & (own_tile_state | TileState.corner | TileState.throne) != 0:
            return False

        # see regular move method for a short explanation
        self.board[move_to] = (self.board[move_to] & TileState.throne) | \
                                 (self.board[move_from] & (TileState.white | TileState.black | TileState.king))
        self.board[move_from] = self.board[move_from] & \
                                     ~(TileState.white | TileState.black | TileState.king)  # remove piece from tile

        board_bytes = self.board.tobytes()
        result = board_bytes in self.board_states_dict and self.board_states_dict[board_bytes] == 2
        self.board[move_from] = previous_from
        self.board[move_to] = previous_to
        return result

    # # reverts the last action and returns the board state to the state before
    # def undo_last_action(self):
    #     if len(self.board_stack) > 0:
    #         if self.board_states_dict[self.board.tobytes()] == 1:
    #             self.board_states_dict.pop(self.board.tobytes())
    #         else:
    #             self.board_states_dict[self.board.tobytes()] -= 1
    #         self.board = np.frombuffer(self.board_stack.pop(), dtype=np.int32).reshape((self.size + 2, self.size + 2))
    #         self.board.setflags(write=1)
    #         self.king_position = self.king_position_stack.pop()
    #         self.turns_without_capture_count = self.turns_without_capture_count_stack.pop()
    #         self.outcome = Outcome.ongoing
    #         self.turn_count -= 1
    #         self.update_board_states()
    #     else:
    #         raise Exception("undo_last_action() failed because there is no action left to revert.")
        # if len(self.action_stack) > 0:
        #     # update board_states_dict
        #     if self.board.tobytes() not in self.board_states_dict:
        #         print(self.board.tobytes())
        #         print(self.board_states_dict)
        #     if self.board_states_dict[self.board.tobytes()] == 1:
        #         self.board_states_dict.pop(self.board.tobytes())
        #     else:
        #         self.board_states_dict[self.board.tobytes()] -= 1
        #
        #     # revert the movement of the piece
        #     (from_position, to_position), tile_state = self.action_stack.pop()
        #     self.board[to_position] = TileState.empty
        #     self.board[from_position] = tile_state
        #     # update king_position if that action was the king being moved
        #     if self.king_position == to_position:
        #         self.king_position = from_position
        #
        #     # revert captures
        #     captured_pieces = self.capture_stack.pop()
        #     other_player_tile_state = TileState.white if tile_state == TileState.black else TileState.black
        #     for position in captured_pieces:
        #         self.board[position] = other_player_tile_state if self.king_position != position else TileState.king
        #     if tile_state == TileState.black:
        #         self.white_pieces += len(captured_pieces)
        #     else:
        #         self.black_pieces += len(captured_pieces)
        #
        #     self.outcome = Outcome.ongoing
        #     self.turn_count -= 1
        #     self.turns_without_capture_count = self.turns_without_capture_count_stack.pop()
        #     self.update_board_states()
        # else:
        #     raise Exception("undo_last_action() failed because there is no action left to revert.")

    def __str__(self):
        return np.array_str(self.board) + str(self.board_states_dict[self.board.tobytes()])

    # bytes are much faster than strings, so use this method if you can
    def bytes(self):
        return self.board[1:self.size + 1, 1: self.size + 1].tostring() \
               + self.board_states_dict[self.board.tobytes()].to_bytes(1, byteorder='big')
