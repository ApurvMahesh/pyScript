import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def predictOutput(new_json_data, test_size = 0.2, k_fold = 5):
    data = pd.read_csv('nba_logreg.csv')
    data_renamed = data.rename(columns = {'GP':'GamesPlayed', 'MIN':'MinutesPlayed', 'PTS':'Points/Game', 'FGM':'FieldGoalsMade', 
                        'FGA':'FieldGoalsAttempts', 'FG%':'FieldGoals%', '3P Made':'3PointMade', '3PA':'3PointAttempts', 
                        '3P%':'3Point%', 'FTM':'FreeThrowMade', 'FTA':'FreeThrowAttempts', 'FT%':'FreeThrow%', 
                        'OREB':'OffensiveRebounds', 'DREB':'DefensiveRebounds', 'REB':'Rebounds', 'AST':'Assists', 
                        'STL':'Steals', 'BLK':'Blocks', 'TOV':'Turnovers'}) 
    data_no_duplicates = data_renamed.drop_duplicates()
    data_no_null = data_no_duplicates.fillna(0)

    # Selecting only numeric columns for calculating skewness
    numeric_data = data_no_duplicates.select_dtypes(include=['number'])

    # Calculating the median of each column excluding missing values
    column_medians = numeric_data.median()

    # Imputing missing values with the corresponding median for each column
    data_imputed_median = data_no_duplicates.fillna(column_medians)

    non_binary_columns = []
    # Iterating through columns and checking for unique values
    for col in data_imputed_median.columns:
        if data_imputed_median[col].nunique() > 2:  # Check if more than 2 unique values
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
        if data_imputed_median[column].dtype != 'object':
            lr, ur = remove_outlier(data_no_null[column])
            data_imputed_median[column] = np.where(data_no_null[column] > ur, ur, data_no_null[column])
            data_imputed_median[column] = np.where(data_no_null[column] < lr, lr, data_no_null[column])

    X = data_imputed_median.drop(columns=['TARGET_5Yrs']) # all columns except the target_class will be stored in as features
    y = data_imputed_median['TARGET_5Yrs']

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
    selected_feature_names_k = X.columns[selected_indices]

    # Selecting the top 10 significant features using Fisher Score 
    f_score_selector = SelectKBest(score_func=f_classif, k=10)
    X_f_score_selected = f_score_selector.fit_transform(X, y)

    # Get the indices of the selected features
    selected_indices = f_score_selector.get_support(indices=True)

    # Print the selected feature names
    selected_feature_names_f = X.columns[selected_indices]

    # SMOTE Oversampling for handling class imbalance
    # Selecting the top significant features for training
    selected_features =  [element for element in selected_feature_names_f if element in selected_feature_names_k]
    X_selected = X[selected_features]

    # Instantiating the SMOTE algorithm
    smote = SMOTE(random_state=42)

    # Applying SMOTE to the selected features and target variable
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = test_size, random_state = 0)

    randomForestMeanScore = "{:.2f}".format(RandomForestCrossvalidation(k_fold, X_train, y_train))
    logisticRegressionMeanScore = "{:.2f}".format(LogisticRegressionCrossvalidation(k_fold, X_train, y_train))
    gradientBoostMeanScore = "{:.2f}".format(GradientBoostCrossvalidation(k_fold, X_train, y_train))
    svmMeanScore = "{:.2f}".format(SupportVectorMachinesCrossvalidation(k_fold, X_train, y_train))

    new_data = pd.DataFrame.from_dict(new_json_data, orient='index').T
    new_data = new_data[selected_features]
    randomForestPrediction = RandomForestPrediction(X_train, y_train, new_data)
    logisticregressionPrediction = LogisticRegressionPrediction(X_train, y_train, new_data)
    gradientBoostPrediction = GradientBoostingPrediction(X_train, y_train, new_data)
    svmPrediction = SupportVectorPrediction(X_train, y_train, new_data)

    return [
        str(k_fold) + "-fold with Train Data Size " + str((1-test_size)*100) +"% & selected features based on Gini & F1 Score - " + str(selected_features),
        "Random Forest with mean score "+str(randomForestMeanScore)+" predicts output as "+ str(randomForestPrediction), 
        "Logistic Regression with mean score "+str(logisticRegressionMeanScore)+" predicts output as "+ str(logisticregressionPrediction),
        "Gradient Boost with mean score "+str(gradientBoostMeanScore)+" predicts output as "+ str(gradientBoostPrediction), 
        "Support Vector Machines with mean score "+str(svmMeanScore)+" predicts output as "+ str(svmPrediction), 
 
        ]

def RandomForestCrossvalidation(k_fold, X_train, y_train):
    # Defining the number of folds (k)
    k = k_fold

    # Initializing the KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    # Initializing Random Forest classifier 
    model = RandomForestClassifier(n_estimators=100, random_state=0)  

    # Performing k-fold cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=kf)

    # The cross-validation scores
    print("Cross-validation scores:", scores)

    # Calculating the mean cross-validation score
    return scores.mean()

def RandomForestPrediction(X_train, y_train, X_test):
  # Initializing Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)

    # Training the model on the entire training dataset
    rf_model.fit(X_train, y_train)

    # Making predictions on the testing dataset
    y_pred = rf_model.predict(X_test)  

    return y_pred

def LogisticRegressionCrossvalidation(k_fold, X_train, y_train):
    # Defining the number of folds (k)
    k = k_fold

    # Initializing the KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    # Initializing model 
    model = LogisticRegression(max_iter=1000)  

    # Performing k-fold cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=kf)

    # The cross-validation scores
    print("Cross-validation scores:", scores)

    # Calculating the mean cross-validation score
    return scores.mean()

def LogisticRegressionPrediction(X_train, y_train, X_test):
  # Initializing Random Forest classifier
    model = LogisticRegression(max_iter=1000)  

    # Training the model on the entire training dataset
    model.fit(X_train, y_train)

    # Making predictions on the testing dataset
    y_pred = model.predict(X_test)  

    return y_pred

def GradientBoostCrossvalidation(k_fold, X_train, y_train):
    # Defining the number of folds (k)
    k = k_fold

    # Initializing the KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    # Initializing Gradient Boosting classifier 
    model = GradientBoostingClassifier(n_estimators=100, random_state=0)  

    # Performing k-fold cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=kf)

    # The cross-validation scores
    print("Cross-validation scores:", scores)

    # Calculating the mean cross-validation score
    return scores.mean()

def GradientBoostingPrediction(X_train, y_train, X_test):
  # Initializing Random Forest classifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=0)  

    # Training the model on the entire training dataset
    model.fit(X_train, y_train)

    # Making predictions on the testing dataset
    y_pred = model.predict(X_test)  

    return y_pred

def SupportVectorMachinesCrossvalidation(k_fold, X_train, y_train):
    # Defining the number of folds (k)
    k = k_fold

    # Initializing the KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    # Initializing Support Vector Machine classifier 
    model = SVC(kernel='rbf', random_state=0)  # Example: using Radial Basis Function (RBF) kernel

    # Performing k-fold cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=kf)

    # The cross-validation scores
    print("Cross-validation scores:", scores)

    # Calculating the mean cross-validation score
    return scores.mean()

def SupportVectorPrediction(X_train, y_train, X_test):
  # Initializing Random Forest classifier
    model = SVC(kernel='rbf', random_state=0)  # Example: using Radial Basis Function (RBF) kernel

    # Training the model on the entire training dataset
    model.fit(X_train, y_train)

    # Making predictions on the testing dataset
    y_pred = model.predict(X_test)  

    return y_pred