import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import csv
import GetTourney as GT
import bracket_visualizer

class FeatureEngineering:
    def __init__(self, reg_season_detailed_results_file = "RegularSeasonDetailedResults.csv"):
        self.data = {}
        filereader = csv.reader(open(reg_season_detailed_results_file), delimiter=",")
        header = filereader.next()

        self.all_games = []
        for game in filereader:
            self.all_games.append(game)

#This for loop goes through each game and creates an item in "data" for each team that has schedule=[] and season=[]
#schedule is append with [opponent, win/loss, team score, opponent's score] for the winning and losing team for each matchup
#season is just [] for now, will be filled with stats later

    def create_features(self):

        #attaching ELO scores to the game results
        # loops through each game in the DetailedResults and
        # keeps track of ELO through each game
        # finally it has the end of reg season ELO for each team for each year
        # also, verify ELO formula
        elo_dict={}
        for game in self.all_games:
            year = game[0]
            if year not in elo_dict:
                elo_dict[year] = {}
            wteam = game[2]
            lteam = game[4]

            if wteam not in elo_dict[year]:
                elo_dict[year][wteam] = 1600
            if lteam not in elo_dict[year]:
                elo_dict[year][lteam] = 1600

            winner_rank = elo_dict[year][wteam]
            loser_rank = elo_dict[year][lteam]

            game.append(winner_rank)
            game.append(loser_rank)

            rank_diff = winner_rank - loser_rank
            exp = (rank_diff * -1) / 400
            odds = 1 / (1 + 10 ** exp)
            if winner_rank < 2100:
                k = 32
            elif winner_rank >= 2100 and winner_rank < 2400:
                k = 24
            else:
                k = 16
            new_winner_rank = round(winner_rank + (k * (1 - odds)))
            new_rank_diff = new_winner_rank - winner_rank
            new_loser_rank = loser_rank - new_rank_diff

            elo_dict[year][wteam] = new_winner_rank
            elo_dict[year][lteam] = new_loser_rank


        # filling in the schedueles for all the teams
        for game in self.all_games:
            if game[0] not in self.data:
                self.data[game[0]]={}
            year = game[0]
            wteam = game[2]
            lteam= game[4]
            if wteam not in self.data[year]:
                self.data[year][wteam] = {'schedule':[], 'season':[], 'season_stats':{}}

            a = [lteam,1, game[3], game[5],game[-2],game[-1]] #to the winning team schedule, append [opponent #, 1(indicating a win),team score, opponent score,team ELO going into game, opponent ELO going into game]
            a = [int(x) for x in a]
            # a.extend(game[8:21])
            self.data[year][wteam]['schedule'].append(a)

            if lteam not in self.data[year]:
                self.data[year][lteam] = {'schedule': [], 'season':[],'season_stats':{}}
            b = [wteam,0, game[5], game[3],game[-1],game[-2]] #to the losing team schedule, append [opponent #, 0(indicating a loss),team score, opponent score,team ELO going into game, opponent ELO going into game]
            b = [int(x) for x in b]
            # b.extend(game[21:14])
            # data[year][wteam]['schedule'].append([wteam,0, game[5]])
            self.data[year][lteam]['schedule'].append(b)




        years = self.data.keys()

        # RPI --------------------------------------------------------------------------------------------
        RPI_dict = {}
        for year in years:
            if year not in RPI_dict:
                RPI_dict[year] = {}
            teams = self.data[year].keys()
            for team in teams:
                if team not in RPI_dict[year]:
                    RPI_dict[year][team] = {'WP': None, 'OWP_list': [], 'OOWP_list': []}
                sche = self.data[year][team]['schedule']
                # print len(sche)
                num_wins = 0
                num_losses = 0
                for game in sche:
                    if game[1] == 1:
                        num_wins += 1
                    else:
                        num_losses += 1

                    op_sche = self.data[year][str(game[0])]['schedule']
                    op_num_wins = 0
                    op_num_losses = 0
                    for op_game in op_sche:
                        if str(op_game[0]) != team:
                            if op_game[1] == 1:
                                op_num_wins += 1
                            else:
                                op_num_losses += 1
                    RPI_dict[year][team]['OWP_list'].append(op_num_wins / float(op_num_wins + op_num_losses))
                RPI_dict[year][team]['WP'] = num_wins / float(num_wins + num_losses)
                RPI_dict[year][team]['OWP'] = sum(RPI_dict[year][team]['OWP_list']) / len(
                    RPI_dict[year][team]['OWP_list'])

        for year in years:
            teams = self.data[year].keys()
            for team in teams:
                sche = self.data[year][team]['schedule']
                for game in sche:
                    RPI_dict[year][team]['OOWP_list'].append(RPI_dict[year][str(game[0])]['OWP'])

                RPI_dict[year][team]['OOWP'] = sum(RPI_dict[year][team]['OOWP_list']) / len(RPI_dict[year][team]['OOWP_list'])
                RPI_dict[year][team]['RPI'] = 0.25*RPI_dict[year][team]['WP'] + 0.5*RPI_dict[year][team]['OWP'] + 0.25*RPI_dict[year][team]['OOWP']

        # print team
        # RPI_dict[year]['win_perc'] =


        # these next 3 nested for loops:
        # 1st(outer loop) goes through each year
        # 2nd(1st inner loop)for each team, calcs their average pts/game and pts against/game and saves in "data" for each team under "season"
        # 3rd(2nd inner loop) goes through teams again and calcs offensive power and defensive suppression for each team and saves in "data" for each team under "season"
        # Offensive power(OP): if you're oppenents collectively on average give up 100pts/ game and you're averaging 103, you have 1.3 offensive power
        # Defensive Suppresion(DS): if you're on average give up 75pts/ game and your opponents collectively on average score 100pts/game, you have .75 defensive suppression
        # OP: higher is better. if >1 then you are scoring more than you're oppenents usually allow
        # DS: lower is better. if >1 then you are allowing your oppenents to score more on you than they usually do
        # Season will be a list of [avg points/gm, avg points allowed/gm, OP, DS, ELO]
        # OP and DS calcs look to be weird. check them.
        for year in years:
            teams = self.data[year].keys()
            for team in teams:
                sche = self.data[year][team]['schedule']
                # print len(sche)
                offtot=0
                deftot=0
                for game in sche:
                    offtot += game[2]
                    deftot += game[3]
                # print team
                self.data[year][team]['season'].append(offtot/float(len(sche)))
                self.data[year][team]['season'].append(deftot / float(len(sche)))
                self.data[year][team]['season_stats']['avg_pts_scored'] = offtot/float(len(sche))
                self.data[year][team]['season_stats']['avg_pts_allowed'] = deftot/float(len(sche))
                # print sche
                # print [sum(x) for x in zip(*data[year][team]['schedule'])]


            for team in teams:
                sche = self.data[year][team]['schedule']
                # print len(sche)
                oppofftot=0
                oppdeftot=0
                reg_season_elo = 0
                for game in sche:
                    opp = game[0]
                    # oppofftot += self.data[year][str(opp)]['season'][0]
                    oppofftot += self.data[year][str(opp)]['season_stats']['avg_pts_scored']
                    # oppdeftot += self.data[year][str(opp)]['season'][1]
                    oppdeftot += self.data[year][str(opp)]['season_stats']['avg_pts_allowed']
                    reg_season_elo = game[-2]
                # print team
                # data[year][team]['season'].append(data[year][team]['season'][0]-(oppdeftot/float(len(sche))))
                # data[year][team]['season'].append(data[year][team]['season'][1]-(oppofftot / float(len(sche))))
                # self.data[year][team]['season'].append(self.data[year][team]['season'][0] / (oppdeftot / float(len(sche))))
                # self.data[year][team]['season'].append(self.data[year][team]['season'][1]/(oppofftot / float(len(sche))))
                # self.data[year][team]['season'].append(reg_season_elo)





                self.data[year][team]['season_stats']['OQ'] = self.data[year][team]['season_stats']['avg_pts_scored'] - (oppdeftot / float(len(sche)))
                self.data[year][team]['season_stats']['DQ'] = (oppofftot / float(len(sche))) - self.data[year][team]['season_stats']['avg_pts_allowed']
                self.data[year][team]['season_stats']['OP'] = self.data[year][team]['season_stats']['avg_pts_scored'] / (oppdeftot / float(len(sche)))
                self.data[year][team]['season_stats']['DS'] = self.data[year][team]['season_stats']['avg_pts_allowed']/(oppofftot / float(len(sche)))
                self.data[year][team]['season_stats']['ASM'] = self.data[year][team]['season_stats']['OQ'] + self.data[year][team]['season_stats']['OQ']
                self.data[year][team]['season_stats']['ELO'] = reg_season_elo
                self.data[year][team]['season_stats']['RPI'] = RPI_dict[year][team]['RPI']


    def GetTrainingData(self,trainyearstart, trainyearend):
        # trainyearstart = 2003
        # trainyearend = 2013
        TS = GT.GetTourneySche_train(trainyearstart, trainyearend)
        team_seeds = GT.Seedings()
        train_data = []
        train_labels=[]
        for game in TS:
            team1 = game[0]
            team2 = game[1]
            trainyear = str(game[-1])

            # OP_team1 = self.data[trainyear][team1]['season'][2]
            # OP_team2 = self.data[trainyear][team2]['season'][2]
            # DS_team1 = self.data[trainyear][team1]['season'][3]
            # DS_team2 = self.data[trainyear][team2]['season'][3]
            # ELO_team1 = self.data[trainyear][team1]['season'][-1]
            # ELO_team2 = self.data[trainyear][team2]['season'][-1]
            # PtsPerGame_team1 = self.data[trainyear][team1]['season'][0]
            # PtsPerGame_team2 = self.data[trainyear][team2]['season'][0]

            OP_team1 = self.data[trainyear][team1]['season_stats']['OP']
            OP_team2 = self.data[trainyear][team2]['season_stats']['OP']
            DS_team1 = self.data[trainyear][team1]['season_stats']['DS']
            DS_team2 = self.data[trainyear][team2]['season_stats']['DS']
            ELO_team1 = self.data[trainyear][team1]['season_stats']['ELO']
            ELO_team2 = self.data[trainyear][team2]['season_stats']['ELO']
            ASM_team1 = self.data[trainyear][team1]['season_stats']['ASM']
            ASM_team2 = self.data[trainyear][team2]['season_stats']['ASM']
            RPI_team1 = self.data[trainyear][team1]['season_stats']['RPI']
            RPI_team2 = self.data[trainyear][team2]['season_stats']['RPI']
            seed_team1 = team_seeds[trainyear][team1]
            seed_team2 = team_seeds[trainyear][team2]
            PtsPerGame_team1 = self.data[trainyear][team1]['season_stats']['avg_pts_scored']
            PtsPerGame_team2 = self.data[trainyear][team2]['season_stats']['avg_pts_allowed']
            # train_data.append([(OP_team1+DS_team1)-(OP_team2+DS_team2), ELO_team1 - Elo_team2])
            # train_data.append([PtsPerGame_team1 * (OP_team1 + (DS_team2 - 1)) - PtsPerGame_team2 * (OP_team2 + (DS_team1 - 1)),ELO_team1 - ELO_team2])
            # train_data.append([ASM_team1-ASM_team2,ELO_team1 - ELO_team2])
            # train_data.append([int(seed_team1)-int(seed_team2),ELO_team1 - ELO_team2,ASM_team1-ASM_team2, RPI_team1-RPI_team2])
            train_data.append([ELO_team1 - ELO_team2,ASM_team1-ASM_team2, RPI_team1-RPI_team2])


            # train_data.append([OP_team1, DS_team1, OP_team2, DS_team2, ELO_team1, ELO_team2])
            train_labels.append(game[-2])
        return np.array(train_data), np.array(train_labels)


    def GetTestData(self,testyear,labels=True, TSpara=None):
        # trainyearstart = 2003
        # trainyearend = 2013
        TS = TSpara
        if TSpara ==None:
            TS = GT.GetTourneySche_test(testyear, None)
        team_seeds = GT.Seedings()
        train_data = []
        train_labels = []
        for game in TS:
            team1 = game[0]
            team2 = game[1]
            trainyear = str(testyear)

            OP_team1 = self.data[trainyear][team1]['season_stats']['OP']
            OP_team2 = self.data[trainyear][team2]['season_stats']['OP']
            DS_team1 = self.data[trainyear][team1]['season_stats']['DS']
            DS_team2 = self.data[trainyear][team2]['season_stats']['DS']
            ELO_team1 = self.data[trainyear][team1]['season_stats']['ELO']
            ELO_team2 = self.data[trainyear][team2]['season_stats']['ELO']
            ASM_team1 = self.data[trainyear][team1]['season_stats']['ASM']
            ASM_team2 = self.data[trainyear][team2]['season_stats']['ASM']
            RPI_team1 = self.data[trainyear][team1]['season_stats']['RPI']
            RPI_team2 = self.data[trainyear][team2]['season_stats']['RPI']
            seed_team1 = team_seeds[trainyear][team1]
            seed_team2 = team_seeds[trainyear][team2]
            PtsPerGame_team1 = self.data[trainyear][team1]['season_stats']['avg_pts_scored']
            PtsPerGame_team2 = self.data[trainyear][team2]['season_stats']['avg_pts_allowed']
            # train_data.append([(OP_team1+DS_team1)-(OP_team2+DS_team2), ELO_team1 - Elo_team2])
            # train_data.append([PtsPerGame_team1 * (OP_team1 + (DS_team2 - 1)) - PtsPerGame_team2 * (OP_team2 + (DS_team1 - 1)),ELO_team1 - ELO_team2])
            # train_data.append([ASM_team1-ASM_team2,ELO_team1 - ELO_team2])
            # train_data.append([int(seed_team1) - int(seed_team2), ELO_team1 - ELO_team2, ASM_team1 - ASM_team2 , RPI_team1-RPI_team2])
            train_data.append([ELO_team1 - ELO_team2, ASM_team1 - ASM_team2 , RPI_team1-RPI_team2])



            if labels == True:
                train_labels.append([game[-2]])

        if labels == True:
            return np.array(train_data), np.array(train_labels)
        return np.array(train_data)

def Train(x,y):
    model1 = MLPClassifier(hidden_layer_sizes=(100, 100))
    model2 = KNeighborsClassifier(n_neighbors=17, weights='uniform', p=2)
    model3 = DecisionTreeClassifier()
    model4 = SVC(probability=True)
    model5 = AdaBoostClassifier()
    # vmodel = VotingClassifier(estimators=[('NN', model1),  ('DT', model3), ('ADA', model5)], voting='soft')
    vmodel = model5
    vmodel.fit(x,y)
    return vmodel


def Test(model, test_x, prob=False):
    if prob== False:
        prediction = model.predict(test_x)
    else:
        prediction = model.predict_proba(test_x)
    return prediction

def grid_search_model(grid_model,parameters):
    for x in range(2003,2015):
        max_n = min((2016-x)*64*2/5,100)
        train_x, train_y = FE.GetTrainingData(x, 2016)
        parameters = {'n_neighbors': range(1, max_n), 'weights': ['uniform', 'distance'], 'p':[1,2]}
        grid_model = model_selection.GridSearchCV(grid_model, parameters, n_jobs=-1, cv=5)
        grid_model.fit(train_x,train_y)
        print x
        print grid_model.best_score_
        grid_model = grid_model.best_estimator_
        print grid_model


if __name__ == '__main__':
    FE = FeatureEngineering()
    FE.create_features()
    train_x, train_y = FE.GetTrainingData(2014,2016)
    test_x, test_y = FE.GetTestData(2016)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    dt = DecisionTreeClassifier()
    dt.fit(train_x,train_y)
    print 'features'
    print dt.feature_importances_
    # for x in range(2003,2015):
    #     max_n = min((2016-x)*64*2/5,100)
    #     train_x, train_y = FE.GetTrainingData(x, 2016)
    #     KNN = KNeighborsClassifier(n_neighbors=40,weights='distance', p=1)
    #     parameters = {'n_neighbors': range(1, max_n), 'weights': ['uniform', 'distance'], 'p':[1,2]}
    #     KNN = model_selection.GridSearchCV(KNN, parameters, n_jobs=-1, cv=5)
    #     KNN.fit(train_x,train_y)
    #     print x
    #     print KNN.best_score_
    #     KNN = KNN.best_estimator_
    #     print KNN

    # grid_search_model(MLPClassifier(),{'hidden_layer_sizes': zip(range(10,100, 10),range(10, 100, 10))})
    grid_search_model(AdaBoostClassifier(),{'n_estimators': range(5,50)})


    model = Train(train_x,train_y)
    # print Test(model, test_x, prob=True)
    winners = {}
    static_seeds= GT.RealSeedings()
    teams = GT.GetTeams()
    for n in range(0,1):
        slots = GT.GetSlots(2017)
        seeds = static_seeds
        for i,game in enumerate(slots):
            year =2017
            team1 = seeds[str(year)][game[1]]
            team2 = seeds[str(year)][game[2]]
            game_test = FE.GetTestData(year,labels=False,TSpara=[[team1, team2]])
            game_test = scaler.transform(game_test)
            chances = Test(model,game_test,prob=True)
            if chances[0][0]> chances[0][1]:
                winner = [team1]
            else:
                winner = [team2]

            # winner = np.random.choice([team1,team2], 1 ,p=chances[0])
            seeds[str(year)][game[0]] = winner[0]


            #Hardcode the First Four games after they happen. Check to make sure the W11, Y16 etc is the right play-in game next year
            if game[0] =='W11':
                #1344= Providence 1425=USC
                seeds[str(year)][game[0]] = '1344'
            if game[0] == 'W16':
                # 1219=MT. St. Mary(High Point)  1309=New Orleans
                seeds[str(year)][game[0]] = '1291'
            if game[0] == 'Y16':
                # 1300=NC Central  1413=UC Davis
                seeds[str(year)][game[0]] = '1300'
            if game[0] == 'Z11':
                # 1243=Kansas St  1448=Wake Forest
                seeds[str(year)][game[0]] = '1243'



            # print teams[winner[0]]
            if i == len(slots)-1:
                if teams[winner[0]] in winners:
                    winners[teams[winner[0]]] += 1
                else:
                    winners[teams[winner[0]]] = 1
        print n
        print seeds

    bracket_visualizer.visualize_bracket(seeds['2017'])

    print_wins=[]
    for  team in winners.keys():
        print_wins.append([team,winners[team]])

    print_wins.sort(key=lambda x: x[1], reverse=True)
    print print_wins
    print winners