import time
from pickle import Unpickler

import Arena
from tafl.TaflGame import TaflGame


filename = "../test_replay.taflreplay"
wait_time = 10  # seconds between moves


class ReplayAgent:
    def __init__(self, filename, wait_time):
        with open(filename, "rb") as f:
            self.moves = Unpickler(f).load()
        self.turn_counter = -1
        self.wait_time = wait_time

    def play(self, board, player):
        time.sleep(self.wait_time)
        self.turn_counter += 1
        return self.moves[self.turn_counter]


# this code is run
g = TaflGame(7)
replay = ReplayAgent(filename, wait_time).play
arena = Arena.Arena(replay, replay, g, replay=True)
arena.playGames(1, profile=False)

