# EECS_731-Project_4
In this project, I implemented few regression model (Linear Regression, Random Forest, Gradient Boost, and Neural Network) to predict the score of each team. Then I also tried to modify the parameters of Random Forest to improve the performance. 

# Project Instruction
NFL, MLB, NBA and Soccer scores
1. Set up a data science project structure in a new git repository in your GitHub account
2. Pick one of the game data sets depending your sports preference
https://github.com/fivethirtyeight/nfl-elo-game     
https://github.com/fivethirtyeight/data/tree/master/mlb-elo     
https://github.com/fivethirtyeight/data/tree/master/nba-carmelo   
https://github.com/fivethirtyeight/data/tree/master/soccer-spi   
3. Load the data set into panda data frames
4. Formulate one or two ideas on how feature engineering would help the data set to establish additional value using exploratory data analysis
5. Build one or more regression models to determine the scores for each team using the other columns as features
6. Document your process and results
7. Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

# Datasets
https://github.com/fivethirtyeight/data/tree/master/mlb-elo   

# Results
As a result, all of the regression model did not give good performance (below 55%), Random Forest is the best model in all four models. Also the performance of Random Forest is increasing while the model become more complex. 

# References
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html    
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html    
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression    
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html     
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html     
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html     
