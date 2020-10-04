import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('Data/mlb_elo.csv')
data = data.drop('playoff', axis=1)
data = data.dropna()

team_encodingColumns = data['team1'].append(data['team2'])
pitcher_encodingColumns = data['pitcher1'].append(data['pitcher2'])

team_encoding = preprocessing.LabelEncoder()
pitcher_encoding = preprocessing.LabelEncoder()
team_encoding.fit(team_encodingColumns)
pitcher_encoding.fit(pitcher_encodingColumns)

data['team1'] = team_encoding.transform(data['team1'])
data['team2'] = team_encoding.transform(data['team2'])
data['pitcher1'] = pitcher_encoding.transform(data['pitcher1'])
data['pitcher2'] = pitcher_encoding.transform(data['pitcher2'])

data = data.drop('date',axis=1)

label = data[['score1','score2']]
sample = data.drop(['score1','score2'],axis=1)
sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size=0.1)

LR = LinearRegression()
LR.fit(sample_train,label_train)
LR_predict = LR.predict(sample_test)
accuracy_LR = r2_score(label_test,LR_predict)

RF = RandomForestRegressor(n_estimators=200)
RF.fit(sample_train,label_train)
RF_predict = RF.predict(sample_test)
accuracy_RF = r2_score(label_test,RF_predict)

GB1 = GradientBoostingRegressor()
GB1.fit(sample_train,label_train['score1'])
GB_predict1 = GB1.predict(sample_test)
accuracy_GB_1 = r2_score(label_test['score1'],GB_predict1)

GB2 = GradientBoostingRegressor()
GB2.fit(sample_train,label_train['score2'])
GB_predict2 = GB2.predict(sample_test)
accuracy_GB_2 = r2_score(label_test['score2'],GB_predict2)

MLP = MLPRegressor(max_iter=500)
MLP.fit(sample_train,label_train)
MLP_predict = MLP.predict(sample_test)
accuracy_MLP = r2_score(label_test,MLP_predict)


