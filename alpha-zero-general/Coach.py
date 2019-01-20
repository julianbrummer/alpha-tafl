import cProfile
from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle

from tafl.TaflBoard import Player, Outcome
from tafl.TaflGame import MovementType


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, white_nnet, black_nnet, args):
        self.game = game
        self.white_nnet = white_nnet
        self.black_nnet = black_nnet
        self.white_pnet = self.white_nnet.__class__(self.game)  # the competitor network
        self.black_pnet = self.black_nnet.__class__(self.game)
        self.args = args
        self.mcts = MCTS(self.game, self.white_nnet, self.black_nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            # print("turn " + str(episodeStep))
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, self.curPlayer, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            if action == 0:
                print(pi)

            board.print_game_over_reason = False
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            board.print_game_over_reason = False

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                # if board.outcome == Outcome.black:
                #     print(" black wins")
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer)), (x[1],)) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        train_black = True

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                if self.args.profile_coach:
                    prof = cProfile.Profile()
                    prof.enable()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.white_nnet, self.black_nnet, self.args)   # reset search tree
                    iterationTrainExamples += self.executeEpisode()
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()
                if self.args.profile_coach:
                    prof.disable()
                    prof.print_stats(sort=2)

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.white_nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp_white.pth.tar')
            self.black_nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp_black.pth.tar')
            self.white_pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_white.pth.tar')
            self.black_pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_black.pth.tar')

            pmcts = MCTS(self.game, self.white_pnet, self.black_pnet, self.args)

            if train_black:
                self.black_nnet.train(trainExamples)
            else:
                self.white_nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.white_nnet, self.black_nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            # originally:                           v<---np.argmax(..................................)
            arena = Arena(lambda board, turn_player: (pmcts.getActionProb(board, turn_player, temp=0)),
                          lambda board, turn_player: (nmcts.getActionProb(board, turn_player, temp=0)),
                          self.game)
            pwins, nwins, draws, pwins_white, pwins_black, nwins_white, nwins_black \
                = arena.playGames(self.args.arenaCompare, self.args.profile_arena)

            print('NEW/PREV WINS (white, black) : (%d,%d) / (%d,%d) ; DRAWS : %d' % (nwins_white, nwins_black, pwins_white, pwins_black, draws))

            if train_black:
                pwins_color = pwins_black
                nwins_color = nwins_black
            else:
                pwins_color = pwins_white
                nwins_color = nwins_white

            if pwins_color+nwins_color > 0 and float(nwins_color)/(pwins_color+nwins_color) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                if train_black:
                    self.black_nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_black.pth.tar')
                else:
                    self.white_nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp_white.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                if train_black:
                    self.black_nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i, Player.black))
                    self.black_nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                    if nwins_black / nwins_white > 0.8:
                        train_black = False
                else:
                    self.white_nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i, Player.white))
                    self.white_nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                    if nwins_white / nwins_black > 0.8:
                        train_black = True

    def getCheckpointFile(self, iteration, player=None):
        return 'checkpoint_' + ('white_' if player == Player.white else 'black_' if player == Player.black else '') + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
