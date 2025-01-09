# Bill Duong
# Uses scikit-learn's Ridge Classifier to train a machine learning model that predicts the outcome of NBA games from the 2016-2022 seasons

import warnings
from pandas.errors import PerformanceWarning
# Suppress specific warnings
warnings.simplefilter(action="ignore", category=PerformanceWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier # Ridge Regression classifier 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Read and clean NBA game data
game_data = pd.read_csv("nba_games.csv", index_col=0)
game_data = game_data.sort_values("date") # Sort by date
game_data = game_data.reset_index(drop=True) # Reset index and remove old index

# Remove unnecessary or duplicate columns 
del game_data["mp.1"]
del game_data["mp_opp.1"]
del game_data["index_opp"]

# Function to add a target column indicating if a team won their next game 
def add_target(team):
    team["target"] = team["won"].shift(-1)
    return team

# Apply the target column to each team separately 
game_data = game_data.groupby("team", group_keys=False).apply(add_target)

# Convert target columns from boolean values to int values 
game_data = game_data.dropna(subset=["target"]) # Remove rows where teams do not have a next game (last game of season)
game_data["target"] = game_data["target"].astype(int, errors="ignore")

# Remove columns with null values
nulls = pd.isnull(game_data).sum()
null_cols = nulls[nulls > 0]
valid_cols = game_data.columns[~game_data.columns.isin(null_cols.index)]
game_data = game_data[valid_cols].copy()

# Initialie the Ridge Classifier and Sequential Feature Selection process
rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split, n_jobs=1)

# Exclude non-predictive columns from model training
removed_cols = ["season", "date", "won", "target", "team", "team_opp"]
selected_cols = game_data.columns[~game_data.columns.isin(removed_cols)]

# Scale the selected features to the range [0, 1] to improve Ridge Regression performance 
scaler = MinMaxScaler()
game_data[selected_cols] = scaler.fit_transform(game_data[selected_cols])

# Identify the best 30 predictive features 
sfs.fit(game_data[selected_cols], game_data["target"])
predictors = list(selected_cols[sfs.get_support()])

# Function to evaluate the performance of a machine learning model over data from multiple NBA seasons
# Trains a model on data from previous seasons and tests it on the current season 
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]

        model.fit(train[predictors], train["target"]) # Fit the model on training data from specified predictors  

        preds = model.predict(test[predictors]) # Make predictions on the test set 
        preds = pd.Series(preds, index=test.index)

        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)

# Add columns to find rolling averages 
game_data_rolling = game_data[list(selected_cols) + ["won", "team", "season"]]

# Function to find each team's average performance over the last n games 
def find_team_averages(team, n_games):
    rolling =  team[selected_cols].rolling(n_games).mean()
    return rolling

# Find each team's average performance over the last n_games for each season 
n_games = 10
game_data_rolling = game_data_rolling.groupby(["team", "season"], group_keys=False).apply(lambda x: find_team_averages(x, n_games))

# Reanme the rolling average columns to reflect the window size (ex. "points_10" for 10-game average)
rolling_cols = [f"{col}_{n_games}" for col in game_data_rolling.columns]
game_data_rolling.columns = rolling_cols 

# Add rolling average data to the main game data 
game_data = pd.concat([game_data, game_data_rolling], axis=1)
game_data = game_data.dropna()

# Function to shift column data to reflect the next game's value
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col 

# Function to add a new column representing the value of a feature in the next game 
def add_col(game_data, col_name):
    return game_data.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

# Add additional columns to indicate next game features
game_data["home_next"] = add_col(game_data, "home")
game_data["team_opp_next"] = add_col(game_data, "team_opp")
game_data["date_next"] = add_col(game_data, "date")

# Merge the game data with opponent's game data 
game_data_with_opponent = game_data.merge(game_data[rolling_cols + ["team_opp_next", "date_next", "team"]], 
                                left_on=["team", "date_next"], 
                                right_on=["team_opp_next", "date_next"],
                                )

removed_cols = list(game_data_with_opponent.columns[game_data_with_opponent.dtypes == "object"]) + removed_cols

selected_cols = game_data_with_opponent.columns[~game_data_with_opponent.columns.isin(removed_cols)]
sfs.fit(game_data_with_opponent[selected_cols], game_data_with_opponent["target"])

predictors = list(selected_cols[sfs.get_support()])

# Perform backtesting using the trained Ridge Classifier model and selected predictors 
predictions = backtest(game_data_with_opponent, rr, predictors)

accuracy = accuracy_score(predictions["actual"], predictions["prediction"])
# Accuracy of Ridge Classfier model on predicting outcome of NBA games 
print(f"Accuracy of the trained Ridge Regression Classifier model: {accuracy:.4f}")