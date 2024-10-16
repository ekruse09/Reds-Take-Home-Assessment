'''
Elijah Kruse
10/14/2024
Reds Trainee Take Home Assessment

This script is designed for a baseball analytics assessment. 
It processes pitch data and builds a machine learning model to predict the type of pitch thrown during a game based on various game and player features. 
The model uses historical data and aims to classify pitch types into three categories: fastballs (FB), breaking balls (BB), and off-speed pitches (OS). 
It uses a Random Forest Regressor, optimizes hyperparameters through GridSearchCV, and evaluates the model's performance using Mean Absolute Error (MAE). 

Inputs:
1. data.csv: A CSV file containing historical pitch data with columns such as player IDs, game stats, and pitch characteristics.
2. predictions.csv: An Excel file to store predictions, where the model writes the predicted pitch type percentages for each batter.

Outputs:
1. Model Performance: The script prints the best parameters for the Random Forest model based on cross-validation (best number of trees).
2. Mean Absolute Error (MAE) score for the model.
3. Prediction Results: An Excel file (predictions.csv) with updated columns showing the predicted pitch type probabilities for each batter.

Side Effects:
1. Data Modifications: The script performs preprocessing on the input CSV, including filtering, dropping irrelevant columns, and applying one-hot encoding to categorical features.
2. File I/O: The script reads from CSV and Excel files, modifies data, and saves results back into an Excel file.
3. Model Training: The script trains a Random Forest model using historical pitch data.
4. Cross-Validation: Hyperparameter optimization is conducted using GridSearchCV, which is computationally expensive.
'''

# Import Block
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_absolute_error

# INPUTS
# File path to data.csv (CHANGE TO YOUR FILE PATH)
data_file_path = r"C:\Users\elija\Downloads\OneDrive_1_10-13-2024\data.csv"
# File path to predictions.csv (CHANGE TO YOUR FILE PATH)
output_file = r"C:\Users\elija\Downloads\OneDrive_1_10-13-2024\predictions.csv"



# DATA PREPROCESSING

# Load the CSV data file into a DataFrame
df = pd.read_csv(data_file_path)

# The given data came with too many columns. Drop irrelevant ones (irrelevant features)
# Some could be considered relevant, but I'll explain my selection process in my technical report
columns_to_drop = [
    'PLAYER_NAME', 'PITCH_NAME', 'GAME_PK', 'GAME_YEAR', 'GAME_DATE',
    'HOME_TEAM', 'AWAY_TEAM', 'INNING_TOPBOT', 'EVENTS',
    'DESCRIPTION', 'TYPE', 'ZONE', 'PLATE_X', 'PLATE_Z',
    'SZ_TOP', 'SZ_BOT', 'BB_TYPE', 'HIT_LOCATION', 'HC_X',
    'HC_Y', 'HIT_DISTANCE_SC', 'LAUNCH_SPEED', 'LAUNCH_ANGLE',
    'ESTIMATED_BA_USING_SPEEDANGLE', 'ESTIMATED_WOBA_USING_SPEEDANGLE',
    'WOBA_VALUE', 'WOBA_DENOM', 'BABIP_VALUE', 'ISO_VALUE',
    'LAUNCH_SPEED_ANGLE', 'HOME_SCORE', 'AWAY_SCORE',
    'POST_AWAY_SCORE', 'POST_HOME_SCORE', 'POST_BAT_SCORE',
    'POST_FLD_SCORE', 'DELTA_HOME_WIN_EXP', 'DELTA_RUN_EXP'
]
df = df.drop(columns=columns_to_drop)

# Drop rows with "missing" data. Again, this will be justified in my technical report
# Drop rows where PITCH_TYPE is 'FA' or 'PO'
df = df[~df['PITCH_TYPE'].isin(['FA', 'PO'])]
df = df[df['PITCH_TYPE'].notna() & (df['PITCH_TYPE'] != '')]

# Map PITCH_TYPE to FB, OS, or BB
pitch_type_mapping = {
    'FC': 'FB', 'FF': 'FB', 'SI': 'FB',
    'CH': 'OS', 'EP': 'OS', 'FO': 'OS', 
    'FS': 'OS', 'KN': 'OS',
    'CS': 'BB', 'CU': 'BB', 'KC': 'BB',
    'SC': 'BB', 'SL': 'BB', 'ST': 'BB', 
    'SV': 'BB'
}
df['PITCH_TYPE'] = df['PITCH_TYPE'].replace(pitch_type_mapping)


# Define valid values for each column (for one hot encoding prep)
valid_values = {
    'PITCH_TYPE': ['FB', 'BB', 'OS'],
    'BAT_SIDE': ['R', 'L'],
    'THROW_SIDE': ['R', 'L'],
    'ON_1B': ['NA'] + [f'{i:06d}' for i in range(100000)],  # Player IDs are 6 digit numbers
    'ON_2B': ['NA'] + [f'{i:06d}' for i in range(100000)],
    'ON_3B': ['NA'] + [f'{i:06d}' for i in range(100000)],
    'IF_FIELDING_ALIGNMENT': ['Standard', 'Strategic', 'Infield shift', 'Infield shade'],
    'OF_FIELDING_ALIGNMENT': ['Standard', '4th outfielder', 'Strategic']
}

# Filter rows where any of the specified features are invalid (drop these rows)
for column, values in valid_values.items():
    df = df[df[column].isin(values) | df[column].isnull()]

# One-hot encoding for specified columns (non numerical, non boolean)
# Create one-hot encodings for PITCH_TYPE, BAT_SIDE, THROW_SIDE, IF_FIELDING_ALIGNMENT, and OF_FIELDING_ALIGNMENT
one_hot_encoded_df = pd.get_dummies(df, columns=[
    'PITCH_TYPE', 'BAT_SIDE', 'THROW_SIDE', 
    'IF_FIELDING_ALIGNMENT', 'OF_FIELDING_ALIGNMENT'
], drop_first=False)

# Create boolean columns for ON_1B, ON_2B, and ON_3B (its PLAYER_ID in data, didn't make sense to keep it here)
one_hot_encoded_df['ON_1B'] = one_hot_encoded_df['ON_1B'].apply(lambda x: True if x != 'NA' else False)
one_hot_encoded_df['ON_2B'] = one_hot_encoded_df['ON_2B'].apply(lambda x: True if x != 'NA' else False)
one_hot_encoded_df['ON_3B'] = one_hot_encoded_df['ON_3B'].apply(lambda x: True if x != 'NA' else False)
df = one_hot_encoded_df



# ML MODELING
# Based off the features, we'll predict what pitch type was thrown.
# Then, we'll average the pitch type predictions by BATTER_ID.

# Features
X = df[['BATTER_ID', 'PITCHER_ID', 'INNING', 'AT_BAT_NUMBER', 'PITCH_NUMBER',
       'OUTS_WHEN_UP', 'BALLS', 'STRIKES', 'ON_1B', 'ON_2B', 'ON_3B',
       'BAT_SCORE', 'FLD_SCORE', 'BAT_SIDE_L', 'BAT_SIDE_R', 'THROW_SIDE_L',
       'THROW_SIDE_R', 'IF_FIELDING_ALIGNMENT_Infield shade',
       'IF_FIELDING_ALIGNMENT_Infield shift', 'IF_FIELDING_ALIGNMENT_Standard',
       'IF_FIELDING_ALIGNMENT_Strategic',
       'OF_FIELDING_ALIGNMENT_4th outfielder',
       'OF_FIELDING_ALIGNMENT_Standard', 'OF_FIELDING_ALIGNMENT_Strategic']]

# Target
y = df[['PITCH_TYPE_BB', 'PITCH_TYPE_FB', 'PITCH_TYPE_OS']]

# Split the data into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize KFold cross-validator for use within GridSearchCV
kf = KFold(n_splits=5, shuffle=True)

# Initialize the Random Forest model
model = RandomForestRegressor()

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [100, 250, 750, 1000]  # Different numbers of trees to try
}

# Use GridSearchCV to perform cross-validation and search for the best n_estimators
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search on the training data (X_train, y_train)
grid_search.fit(X_train, y_train)

# Output the best parameters (number of trees)
print("Best n_estimators: ", grid_search.best_params_)

# Best model after cross-validation
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)

# Calculate performance metric (mean squared error)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test Set Mean Absolute Error: {mae}")

# Convert predictions to DataFrame and normalize to get percentages
predictions_df = pd.DataFrame(y_pred, columns=['PITCH_TYPE_BB', 'PITCH_TYPE_FB', 'PITCH_TYPE_OS'])
predictions_df = predictions_df.div(predictions_df.sum(axis=1), axis=0)

# Combine BATTER_ID from the test set using the indices
predictions_df['BATTER_ID'] = df.loc[X_test.index, 'BATTER_ID'].reset_index(drop=True)

# Combine predictions by averaging for each BATTER_ID
averaged_predictions_df = predictions_df.groupby('BATTER_ID').mean().reset_index()


# FINAL OUTPUT

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(output_file)

# Define the column names you're looking for
pitch_type_fb_col = "PITCH_TYPE_FB"
pitch_type_bb_col = "PITCH_TYPE_BB"
pitch_type_os_col = "PITCH_TYPE_OS"
batter_id_col = "BATTER_ID"

# Ensure all required columns are present
required_columns = [pitch_type_fb_col, pitch_type_bb_col, pitch_type_os_col, batter_id_col]
if not all(col in df.columns for col in required_columns):
    raise ValueError("One or more required columns not found in the CSV file.")

# Iterate through each row in the DataFrame and match BATTER_ID
for idx, row in df.iterrows():
    batter_id = row[batter_id_col]
    
    # Check if this BATTER_ID exists in the averaged_predictions_df
    if batter_id in averaged_predictions_df['BATTER_ID'].values:
        # Get the predicted percentages for this BATTER_ID
        predicted_row = averaged_predictions_df[averaged_predictions_df['BATTER_ID'] == batter_id].iloc[0]
        predicted_fb = predicted_row['PITCH_TYPE_FB']
        predicted_bb = predicted_row['PITCH_TYPE_BB']
        predicted_os = predicted_row['PITCH_TYPE_OS']
        
        # Update the values in the DataFrame
        df.at[idx, pitch_type_fb_col] = predicted_fb
        df.at[idx, pitch_type_bb_col] = predicted_bb
        df.at[idx, pitch_type_os_col] = predicted_os

# Save the updated DataFrame back to a CSV file
df.to_csv('predictions.csv', index=False)
