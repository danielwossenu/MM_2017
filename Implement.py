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
data = {}
filereader = csv.reader(open("RegularSeasonDetailedResults.csv"), delimiter=",")
header = filereader.next()

#This for loop goes through each game and creates an item in "data" for each team that has schedule=[] and season=[]
#schedule is append with [opponent, win/loss, team score, opponent's score] for the winning and losing team for each matchup
#season is just [] for now, will be filled with stats later


all_games = []
for game in filereader:
    all_games.append(game)


#attaching ELO scores to the game results
elo_dict={}
for game in all_games:
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
for game in all_games:
    if game[0] not in data:
        data[game[0]]={}
    year = game[0]
    wteam = game[2]
    lteam= game[4]
    if wteam not in data[year]:
        data[year][wteam] = {'schedule':[], 'season':[]}

    a = [lteam,1, game[3], game[5],game[-2],game[-1]]
    a = [int(x) for x in a]
    # a.extend(game[8:21])
    data[year][wteam]['schedule'].append(a)

    if lteam not in data[year]:
        data[year][lteam] = {'schedule': [], 'season':[]}
    b = [wteam,0, game[5], game[3],game[-1],game[-2]]
    b = [int(x) for x in b]
    # b.extend(game[21:14])
    # data[year][wteam]['schedule'].append([wteam,0, game[5]])
    data[year][lteam]['schedule'].append(b)



#these next 3 nested for loops:
# 1st(outer loop) goes through each year
# 2nd(1st inner loop)for each team, calcs their average pts/game and pts against/game and saves in "data" for each team under "season"
# 3rd(2nd inner loop) goes through teams again and calcs offensive power and defensive suppression for each team and saves in "data" for each team under "season"
#Offensive power(OP): if you're oppenents collectively on average give up 100pts/ game and you're averaging 103, you have 1.3 offensive power
#Defensive Suppresion(DS): if you're on average give up 75pts/ game and your opponents collectively on average score 100pts/game, you have .75 defensive suppression
#OP: higher is better. if >1 then you are scoring more than you're oppenents usually allow
#DS: lower is better. if >1 then you are allowing your oppenents to score more on you than they usually do
years = data.keys()
for year in years:
    teams = data[year].keys()
    for team in teams:
        sche = data[year][team]['schedule']
        # print len(sche)
        offtot=0
        deftot=0
        for game in sche:
            offtot += game[2]
            deftot += game[3]
        # print team
        data[year][team]['season'].append(offtot/float(len(sche)))
        data[year][team]['season'].append(deftot / float(len(sche)))
        # print sche
        # print [sum(x) for x in zip(*data[year][team]['schedule'])]


    for team in teams:
        sche = data[year][team]['schedule']
        # print len(sche)
        oppofftot=0
        oppdeftot=0
        reg_season_elo = 0
        for game in sche:
            opp = game[0]
            oppofftot += data[year][str(opp)]['season'][0]
            oppdeftot += data[year][str(opp)]['season'][1]
            reg_season_elo = game[-2]
        # print team
        data[year][team]['season'].append(data[year][team]['season'][0]/(oppdeftot/float(len(sche))))
        data[year][team]['season'].append(data[year][team]['season'][1]/(oppofftot / float(len(sche))))
        data[year][team]['season'].append(reg_season_elo)


def GetTrainingData(trainyearstart, trainyearend):
    # trainyearstart = 2003
    # trainyearend = 2013
    TS = GT.GetTourneySche_train(trainyearstart, trainyearend)
    train_data = []
    train_labels=[]
    for game in TS:
        team1 = game[0]
        team2 = game[1]
        trainyear = str(game[-1])

        OP_team1 = data[trainyear][team1]['season'][2]
        OP_team2 = data[trainyear][team2]['season'][2]
        DS_team1 = data[trainyear][team1]['season'][3]
        DS_team2 = data[trainyear][team2]['season'][3]
        ELO_team1 = data[trainyear][team1]['season'][-1]
        Elo_team2 = data[trainyear][team2]['season'][-1]
        PtsPerGame_team1 = data[trainyear][team1]['season'][0]
        PtsPerGame_team2 = data[trainyear][team2]['season'][0]
        train_data.append([PtsPerGame_team1 * (OP_team1 + (DS_team2 - 1)) - PtsPerGame_team2 * (OP_team2 + (DS_team1 - 1)), ELO_team1 - Elo_team2])
        train_labels.append(game[-2])
    return np.array(train_data), np.array(train_labels)


def GetTestData(testyear,labels=True, TSpara=None):
    # trainyearstart = 2003
    # trainyearend = 2013
    TS = TSpara
    if TSpara ==None:
        TS = GT.GetTourneySche_test(testyear, None)
    train_data = []
    train_labels = []
    for game in TS:
        team1 = game[0]
        team2 = game[1]
        trainyear = str(testyear)

        OP_team1 = data[trainyear][team1]['season'][2]
        OP_team2 = data[trainyear][team2]['season'][2]
        DS_team1 = data[trainyear][team1]['season'][3]
        DS_team2 = data[trainyear][team2]['season'][3]
        ELO_team1 = data[trainyear][team1]['season'][-1]
        ELO_team2 = data[trainyear][team2]['season'][-1]
        PtsPerGame_team1 = data[trainyear][team1]['season'][0]
        PtsPerGame_team2 = data[trainyear][team2]['season'][0]
        train_data.append([PtsPerGame_team1 * (OP_team1 + (DS_team2 - 1)) - PtsPerGame_team2 * (OP_team2 + (DS_team1 - 1)),ELO_team1 - ELO_team2])
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
    vmodel = VotingClassifier(estimators=[('NN', model1),  ('DT', model3), ('ADA', model5)], voting='soft')
    vmodel = model2
    vmodel.fit(x,y)
    return vmodel


def Test(model, test_x, prob=False):
    if prob== False:
        prediction = model.predict(test_x)
    else:
        prediction = model.predict_proba(test_x)
    return prediction


train_x, train_y = GetTrainingData(2015,2016)
test_x, test_y = GetTestData(2016)

grid = False
if grid==True:
    for x in range(2013,2016):
        max_n = min((2016-x)*64*2/5,100)
        train_x, train_y = GetTrainingData(x, 2016)
        KNN = KNeighborsClassifier(n_neighbors=40,weights='distance', p=1)
        parameters = {'n_neighbors': range(1, max_n), 'weights': ['uniform', 'distance'], 'p':[1,2]}
        KNN = model_selection.GridSearchCV(KNN, parameters, n_jobs=1, cv=5)
        KNN.fit(train_x,train_y)
        print x
        print KNN.best_score_
        KNN = KNN.best_estimator_
        print KNN

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
        game_test = GetTestData(year,labels=False,TSpara=[[team1, team2]])
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
            seeds[str(year)][game[0]] = '1425'
        if game[0] == 'W16':
            # 1219=MT. St. Mary(High Point)  1309=New Orleans
            seeds[str(year)][game[0]] = '1219'
        if game[0] == 'Y16':
            # 1300=NC Central  1413=UC Davis
            seeds[str(year)][game[0]] = '1413'
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

print_wins=[]
for  team in winners.keys():
    print_wins.append([team,winners[team]])

print_wins.sort(key=lambda x: x[1], reverse=True)
print print_wins
print winners