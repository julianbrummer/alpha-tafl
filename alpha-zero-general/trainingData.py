import pickle
class TrainingData:
#reads the data and removes all the games which do not clearly show a winner
    def readData():
        training_data = pickle.load(open("full_game_stats.p", "rb"))
        while 'ongoing' in training_data['outcome']:
            ind=training_data['outcome'].index('ongoing')
            training_data['outcome'].pop(ind)
            training_data['games'].pop(ind)
            training_data['number_of_moves'].pop(ind)
        for i in training_data['games']:
            if 'resigned' in i:
                ind=training_data['games'].index(i)
                training_data['outcome'].pop(ind)
                training_data['games'].pop(ind)
                training_data['number_of_moves'].pop(ind)
            elif 'timeout' in i:
                ind = training_data['games'].index(i)
                training_data['outcome'].pop(ind)
                training_data['games'].pop(ind)
                training_data['number_of_moves'].pop(ind)
        #print (training_data)
        return training_data
    #returns the translated data in the form: from_x to_x - to_x to_y
    #divides the games by who is the winner so we can train black and white independently
    #trainWhite is a bool which determines which set of games is returned
    #TODO: invoke the data in  the training process
    def translateData(training_data,trainWhite):
        whiteWinnerData={}
        whiteWinnerData['outcome'] = []
        whiteWinnerData['games']=[]
        whiteWinnerData['number_of_moves'] = []
        blackWinnerData = {}
        blackWinnerData['outcome'] = []
        blackWinnerData['games'] = []
        blackWinnerData['number_of_moves'] = []
        for i in training_data['outcome']:
            if i=='white won':
                ind = training_data['outcome'].index(i)
                whiteWinnerData['outcome'].append(training_data['outcome'][ind])
                whiteWinnerData['games'].append(training_data['games'][ind])
                whiteWinnerData['number_of_moves'].append(training_data['number_of_moves'][ind])
            else:
                ind = training_data['outcome'].index(i)
                blackWinnerData['outcome'].append(training_data['outcome'][ind])
                blackWinnerData['games'].append(training_data['games'][ind])
                blackWinnerData['number_of_moves'].append(training_data['number_of_moves'][ind])
        repl = {}
        repl['a'] = '1'
        repl['b'] = '2'
        repl['c'] = '3'
        repl['d'] = '4'
        repl['e'] = '5'
        repl['f'] = '6'
        repl['g'] = '7'
        repl['h'] = '8'
        if trainWhite:
            train_data=whiteWinnerData
        else:
            train_data=blackWinnerData
            #print(train_data)
        for game in train_data['games']:
            gameind = train_data['games'].index(game)
            for draw in game:
                drawind = train_data['games'][gameind].index(draw)
                draw=draw.split("x")[0]
                for item in draw:
                    if item in repl.keys():
                        # look up and replace
                        draw = draw.replace(item, repl[item])
                train_data['games'][gameind][drawind]=draw
        print(train_data)
        return train_data

