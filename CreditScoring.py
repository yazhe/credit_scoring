import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import Imputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt

'''
Log transformation of a column
Inpu: a data frame, original column name， transformed new column name
Output: data frame with new transpormed dataframe
'''
def log_transform(data, column_name, log_column_name):
    data[log_column_name] = data[column_name].apply(np.log)
    data.loc[data[log_column_name] == -np.inf, column_name] = 0
    return data
'''
Feature extraction (Manully explore possible features)
Input: a data frame
Output: a new data frame with selected features for the prediction task
'''
def data_transform(data):
    # data['OldAge'] = data['age'] > 65
    # data['OldAge'] = data['OldAge'].astype(int)  # identify the person reached retirement
    # data['AgeBy10Years'] = data['age'] //
    data = log_transform(data, 'age', 'LogAge')
    data.loc[data.age == 0, 'LogAge'] = -1  # indicating missing age value

    data['LogFamilySize'] = data['NumberOfDependents'].apply(np.log1p)  # family size = 1+#_of_dependents
    data.loc[data.NumberOfDependents.isnull(), 'LogFamilySize'] = -1  # indicating missing value

    # data['NoIncome'] = data['MonthlyIncome'] == 0
    # data['NoIncome'] = data['NoIncome'].astype(int)  # indicating zero income
    data = log_transform(data, 'MonthlyIncome', 'LogMonthlyIncome')
    data.loc[data.MonthlyIncome.isnull(), 'LogMonthlyIncome'] = -1  # indicating missing value

    data['NumberOfOpenCreditLines'] = data['NumberOfOpenCreditLinesAndLoans']-data['NumberRealEstateLoansOrLines']
    data['HasOpenCreditLinesAndLoans'] = data['NumberOfOpenCreditLinesAndLoans'] > 0
    data['HasOpenCreditLinesAndLoans'] = data['HasOpenCreditLinesAndLoans'].astype(int)
    data = log_transform(data, 'NumberOfOpenCreditLinesAndLoans', 'LogOpenCreditLinesAndLoans')

    data['HasRealEstateLoansOrLines'] = data['NumberRealEstateLoansOrLines'] > 0
    data['HasRealEstateLoansOrLines'] = data['HasRealEstateLoansOrLines'].astype(int)
    data = log_transform(data, 'NumberRealEstateLoansOrLines', 'LogRealEstateLoansOrLines')

    # data['HasOpenCreditLines'] = data['NumberOfOpenCreditLines'] > 0
    # data['HasOpenCreditLines'] = data['HasOpenCreditLines'].astype(int)
    data = log_transform(data, 'NumberOfOpenCreditLines', 'LogOpenCreditLines')

    data['NoNumberOfTime30-59DaysPastDueNotWorse'] = data['NumberOfTime30-59DaysPastDueNotWorse'] == 0
    data['NoNumberOfTime30-59DaysPastDueNotWorse'] = data['NoNumberOfTime30-59DaysPastDueNotWorse'].astype(int)
    data = log_transform(data, 'NumberOfTime30-59DaysPastDueNotWorse', 'LogNumberOfTime30-59DaysPastDueNotWorse')
    # indicating abnormal value
    data.loc[data['NumberOfTime30-59DaysPastDueNotWorse'] > 95, 'LogNumberOfTime30-59DaysPastDueNotWorse'] = -1

    data['NoNumberOfTime60-89DaysPastDueNotWorse'] = data['NumberOfTime60-89DaysPastDueNotWorse'] == 0
    data['NoNumberOfTime60-89DaysPastDueNotWorse'] = data['NoNumberOfTime60-89DaysPastDueNotWorse'].astype(int)
    data = log_transform(data, 'NumberOfTime60-89DaysPastDueNotWorse', 'LogNumberOfTime60-89DaysPastDueNotWorse')
    # indicating abnormal value
    data.loc[data['NumberOfTime60-89DaysPastDueNotWorse'] > 95, 'LogNumberOfTime60-89DaysPastDueNotWorse'] = -1

    data['NoNumberOfTimes90DaysLate'] = data['NumberOfTimes90DaysLate'] == 0
    data['NoNumberOfTimes90DaysLate'] = data['NoNumberOfTimes90DaysLate'].astype(int)
    data = log_transform(data, 'NumberOfTimes90DaysLate', 'LogNumberOfTimes90DaysLate')
    data.loc[data['NumberOfTimes90DaysLate'] > 95, 'LogNumberOfTimes90DaysLate'] = -1  # indicating abnormal value

    data['TotalPastDue'] = data['NumberOfTime30-59DaysPastDueNotWorse'] \
        + data['NumberOfTime60-89DaysPastDueNotWorse'] + data['NumberOfTimes90DaysLate']
    data['NoPastDue'] = data['TotalPastDue'] == 0
    data['NoPastDue'] = data['NoPastDue'].astype(int)
    data = log_transform(data, 'TotalPastDue', 'LogPastDue')
    data.loc[data['TotalPastDue'] > 95, 'LogPastDue'] = -1  # indicating abnormal value

    # data['NoDebt'] = data['DebtRatio'] == 0
    # data['NoDebt'] = data['NoDebt'].astype(int)
    # data['WiredDebtRatio'] = data['DebtRatio'] > 10
    # data['WiredDebtRatio'] = data['WiredDebtRatio'].astype(int)  # indicating unreasonably large debt ratio
    data = log_transform(data, 'DebtRatio', 'LogDebtRatio')

    data['NoRevolvingUtilization'] = data['RevolvingUtilizationOfUnsecuredLines'] == 0
    data['NoRevolvingUtilization'] = data['NoRevolvingUtilization'].astype(int)
    # data['WiredRevolvingUtilization'] = data['RevolvingUtilizationOfUnsecuredLines'] > 10
    # indicating unreasonably large debt ratio
    # data['WiredRevolvingUtilization'] = data['WiredRevolvingUtilization'].astype(int)
    data = log_transform(data, 'RevolvingUtilizationOfUnsecuredLines', 'LogRevolvingUtilization')

    # manully created features by dividen/multiply
    data['LogIncomePerPerson'] = data['LogMonthlyIncome'] - data['LogFamilySize']
    data['LogIncomeByAge'] = data['LogMonthlyIncome'] - data['LogAge']
    data['LogDependentsByAge'] = data['LogFamilySize'] - data['LogAge']
    data['LogCreditlines&loanPerPerson'] = data['LogOpenCreditLinesAndLoans'] - data['LogFamilySize']
    data['LogCreditlinesPerPerson'] = data['LogOpenCreditLines'] - data['LogFamilySize']
    data['LogLoanPerPerson'] = data['LogRealEstateLoansOrLines'] - data['LogFamilySize']
    data['LogIncomeByCreditlines&loan'] = data['LogMonthlyIncome'] - data['LogOpenCreditLinesAndLoans']
    data['LogIncomeByCreditlines'] = data['LogMonthlyIncome'] - data['LogOpenCreditLines']
    data['LogIncomeByLoan'] = data['LogMonthlyIncome'] - data['LogRealEstateLoansOrLines']
    data['LogTotalPassDuePerLines'] = data['LogPastDue'] - data['LogOpenCreditLinesAndLoans']
    data['Log30-59PassDuePerLines'] = data['LogNumberOfTime30-59DaysPastDueNotWorse'] \
        - data['LogOpenCreditLinesAndLoans']
    data['Log60-89PassDuePerLines'] = data['LogNumberOfTime60-89DaysPastDueNotWorse']\
        - data['LogOpenCreditLinesAndLoans']
    data['Log90PassDuePerLines'] = data['LogNumberOfTimes90DaysLate'] - data['LogOpenCreditLinesAndLoans']

    data['Debt'] = data['DebtRatio'] * data['MonthlyIncome']
    data.loc[data['Debt'].isnull(), 'Debt'] = 0
    data['LogDebt'] = data['Debt'].apply(np.log)
    data.loc[data.LogDebt == -np.inf, 'LogDebt'] = 0
    data['LogDebtPerLine'] = data['LogDebt'] - data['LogOpenCreditLinesAndLoans']

    data['RemainingIncome'] = data['MonthlyIncome'] - data['Debt']
    data.loc[data['RemainingIncome'].isnull(), 'RemainingIncome'] = 0
    data['LogRemainingIncome'] = data['RemainingIncome'].apply(np.log)
    data.loc[data.LogRemainingIncome == -np.inf, 'LogRemainingIncome'] = 0
    data.loc[data.LogRemainingIncome.isnull(), 'LogRemainingIncome'] = 0
    data = data.drop(['RevolvingUtilizationOfUnsecuredLines', 'age',
                      'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                      'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                      'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                      'NumberOfDependents', 'NumberOfOpenCreditLines',
                      'TotalPastDue', 'Debt', 'RemainingIncome'], axis=1)
    return data

'''
K-fold cross validation to select the best model hyper-parameters
Input: training data features, traing data labels
Output: print the best training score, and best hyper-parameter to console 
'''
def model_selection(train_x, train_y, k):
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', silent=0, scale_pos_weight=13)
    param_dic = {
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 8, 10],
        'min_child_weight': [40,50, 55, 60],
        'n_estimators': [20, 50, 80, 100],
        'subsample': [0.4, 0.5, 0.8, 1],
        'colsample_bytree': [0.4, 0.5, 0.8],
        'max_delta_step': [1.5, 1.8,  2]
    }
    gs = GridSearchCV(model, param_grid=param_dic, cv=k, n_jobs=-1)
    gs.fit(train_x, train_y)
    print("best train score:")
    print(gs.best_score_)
    print("best xgboost parameters: ")
    print(gs.best_params_)

'''
Use a pretrianed model to generate testing results
Input: test data file name, pickle file name of the pretrained model, test result data file name
Output: test result file with the required format for Kaggle submission
'''
def model_testing(test_fname, model_fname, rlt_fname):
    # read model and testing data
    model = joblib.load(model_fname)
    data_test = pd.read_csv(test_fname, sep=',', index_col=0)
    print(data_test.shape, " data dimention")
    data_test = data_transform(data_test)
    print(data_test.shape, " transformed data dimention")
    data_test = data_test.drop(['SeriousDlqin2yrs'], axis=1)
    result = model.predict_proba(data_test)
    result = result[:,1]
    dt = pd.DataFrame(data=result)
    dt.index += 1
    dt.columns = ['Probability']
    dt.to_csv(rlt_fname, index=True, index_label='Id')

def main():
    # read training data
    data = pd.read_csv("./data/cs-training.csv", sep=',', index_col=0)
    print(data.shape, " data dimention")
    data = data_transform(data)
    print(data.shape, " transformed data dimention")

    train_y = data['SeriousDlqin2yrs']
    train_x = data.drop(['SeriousDlqin2yrs'], axis=1)

    # find the best parameter (run this part first to get the best hyper parameter)
    # model_selection(train_x, train_y, 5)

    # train a model with the best parameters
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', silent=1, scale_pos_weight=13, eta=0.3,
                               max_depth=8, min_child_weight=60, n_estimators=26, subsample=0.8, colsample_bytree=0.4,
                               max_delta_step=1.79, gamma=0)
    model.fit(train_x,train_y)
    print("finished training")
    # analyze features
    name = np.array(train_x.columns)
    features = pd.DataFrame({'feature':name, 'score':model.feature_importances_})
    print(features.sort_values(by=['score'], ascending=False))
    xgb.plot_importance(model)  # show the importance of the features
    plt.show()
    joblib.dump(model, './data/model_xgb_current_best.plk')

    # generating test result
    model_testing('./data/cs-test.csv', './data/model_xgb_current_best.plk', './data/test_xgb_current_best.csv')


if __name__ == "__main__":
    main()
