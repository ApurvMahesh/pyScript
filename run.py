import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import jsonify

def predictOutput(new_json_data):
    data = pd.read_csv('nba_logreg.csv')
    data_types = data.dtypes
    data.rename(columns = {'GP':'GamesPlayed', 'MIN':'MinutesPlayed', 'PTS':'Points/Game', 'FGM':'FieldGoalsMade', 
                        'FGA':'FieldGoalsAttempts', 'FG%':'FieldGoals%', '3P Made':'3PointMade', '3PA':'3PointAttempts', 
                        '3P%':'3Point%', 'FTM':'FreeThrowMade', 'FTA':'FreeThrowAttempts', 'FT%':'FreeThrow%', 
                        'OREB':'OffensiveRebounds', 'DREB':'DefensiveRebounds', 'REB':'Rebounds', 'AST':'Assists', 
                        'STL':'Steals', 'BLK':'Blocks', 'TOV':'Turnovers'}, inplace = True) 
    data_no_duplicates = data.drop_duplicates()
    data_no_null = data_no_duplicates.fillna(0)

    non_binary_columns = []
    # Iterating through columns and checking for unique values
    for col in data_no_null.columns:
        if data_no_null[col].nunique() > 2:  # Check if more than 2 unique values
            non_binary_columns.append(col)

    def remove_outlier(col):
        sorted_col = sorted(col)
        Q1, Q3 = np.percentile(sorted_col, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        return lower_bound, upper_bound

    columns_to_remove_outliers_from = ['GamesPlayed', 'MinutesPlayed', 'Points/Game', 'FieldGoalsMade', 'FieldGoalsAttempts', 
                                    'FieldGoals%', '3PointMade', '3PointAttempts', '3Point%', 'FreeThrowMade', 
                                    'FreeThrowAttempts', 'FreeThrow%', 'OffensiveRebounds', 'DefensiveRebounds', 
                                    'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers']

    for column in columns_to_remove_outliers_from:
        if data_no_null[column].dtype != 'object':
            lr, ur = remove_outlier(data_no_null[column])
            data_no_null[column] = np.where(data_no_null[column] > ur, ur, data_no_null[column])
            data_no_null[column] = np.where(data_no_null[column] < lr, lr, data_no_null[column])

    result = data_no_null.groupby('TARGET_5Yrs')['Name'].count().reset_index()

    X = data_no_null.drop(columns=['TARGET_5Yrs']) # all columns except the target_class will be stored in as features
    y = data_no_null['TARGET_5Yrs']

    X = X.drop(columns = ['Name'])
    X.head()

    # Creating a DecisionTreeClassifier model
    model = DecisionTreeClassifier()

    # Fitting the model to the data
    model.fit(X, y)

    # Getting feature importance using Gini index
    feature_importances = model.feature_importances_

    # Creating a SelectKBest object using Gini index as the score function
    k_best_selector = SelectKBest(score_func=lambda X, y: feature_importances, k=10)
    X_k_best_selected = k_best_selector.fit_transform(X, y)

    # Getting the indices of the selected features
    selected_indices = k_best_selector.get_support(indices=True)

    # Printing the selected feature names
    selected_feature_names = X.columns[selected_indices]

    # Selecting the top 10 significant features using Fisher Score 
    f_score_selector = SelectKBest(score_func=f_classif, k=10)
    X_f_score_selected = f_score_selector.fit_transform(X, y)

    # Get the indices of the selected features
    selected_indices = f_score_selector.get_support(indices=True)

    # Print the selected feature names
    selected_feature_names = X.columns[selected_indices]

    # Case 1: 80% training, 20% testing
    # random_seed 42 is being used in our solutions, can be any number - it matters for the reason we need fixed results everytime 
    x_train1, x_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

    # Applying SMOTE oversampling technique to the training data for Case 1
    smote = SMOTE(random_state=42)
    x_train_resampled1, y_train_resampled1 = smote.fit_resample(x_train1, y_train1)

    # Initializing the Decision Tree Classifier for Case 1
    model1 = DecisionTreeClassifier(random_state=42)

    # Training the model for Case 1
    model1.fit(x_train_resampled1, y_train_resampled1)

    # Making predictions for Case 1
    y_pred1 = model1.predict(x_test1)

    # Evaluating the model for Case 1
    print("Classification Report for Case 1 (80% training, 20% testing):")
    print(classification_report(y_test1, y_pred1))

    # Case 2: 10% training, 90% testing
    x_train2, x_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.1, random_state=42)

    # Applying SMOTE oversampling technique to the training data for Case 2
    smote = SMOTE(random_state=42)
    x_train_resampled2, y_train_resampled2 = smote.fit_resample(x_train2, y_train2)

    # Initializing the Decision Tree Classifier for Case 2
    model2 = DecisionTreeClassifier(random_state=42)

    # Training the model for Case 2
    model2.fit(x_train_resampled2, y_train_resampled2)

    # Making predictions for Case 2
    y_pred2 = model2.predict(x_test2)

    # Evaluating the model for Case 2
    print("\nClassification Report for Case 2 (90% training, 10% testing):")
    print(classification_report(y_test2, y_pred2))

    # Initialize the Random Forest Classifier
    rf_model_1 = RandomForestClassifier(random_state=42)
    rf_model_2 = RandomForestClassifier(random_state=42)

    # Train the model
    rf_model_1.fit(x_train_resampled1, y_train_resampled1)
    rf_model_2.fit(x_train_resampled2, y_train_resampled2)


    # Make predictions
    y_pred_rf_1 = rf_model_1.predict(x_test1)
    y_pred_rf_2 = rf_model_2.predict(x_test2)

    print("predicted Y for test 1", y_pred_rf_1)
    print("predicted Y for test 2", y_pred_rf_2)



    # Ensure the columns are in the correct format and order
    # new_data = new_data[X.columns]
    new_data = pd.DataFrame.from_dict(new_json_data, orient='index').T

    # # Predict using Decision Tree Classifier
    prediction_dt = model1.predict(new_data)

    # # Predict using Random Forest Classifier
    prediction_rf = rf_model_1.predict(new_data)

    # # Print the predictions
    # print("Prediction using Decision Tree Classifier:", prediction_dt)
    # print("Prediction using Random Forest Classifier:", prediction_rf)
    return [
        "Result for Prediction using Decision Tree Classifier: " + str(prediction_dt[0]), 
        "Result for Prediction using Random Forest Classifier: " + str(prediction_rf[0])
        ]