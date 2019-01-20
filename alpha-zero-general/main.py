from tafl.pytorch.NNet import NNetWrapper as nn
from Coach import Coach
from tafl.TaflGame import TaflGame
from utils import dotdict

args = dotdict({
    'numIters': 1000,
    'numEps': 50,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,

    'load_folder_file_white': ('/dev/models/8x100x50','best_white.pth.tar'),
    'load_folder_file_black': ('/dev/models/8x100x50','best_black.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'train_other_network_threshold': 1,    # compared with (network that is currently trained wins)/(other network wins)
                                           # toggles the network being trained when threshold is reached

    'profile_coach': False,
    'profile_arena': True,
})

if __name__=="__main__":
    #  g = OthelloGame(6)
    g = TaflGame(7)
    white_nnet = nn(g)
    black_nnet = nn(g)

    # TODO: the load folder file might need to be adjusted to work with two networks. That part is still from when we had only one network
    if args.load_model:
        white_nnet.load_checkpoint(args.load_folder_file_white[0], args.load_folder_file_white[1])
        black_nnet.load_checkpoint(args.load_folder_file_black[0], args.load_folder_file_black[1])

    c = Coach(g, white_nnet, black_nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
