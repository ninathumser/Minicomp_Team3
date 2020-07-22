import os
import time
import random

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb


from joblib import dump, load

#pipeline#

def data_clean():
        
    #### DELETE WHEN MATIAS IS DONE WITH DATA CLEANING###
    data_clean = pd.read_pickle('./data/clean_data_old.pkl')
        
    #### DELETE WHEN MATIAS IS DONE WITH DATA CLEANING###
    return data_clean




def feature_engineering(df):
    df = df.copy()
    
    #convert state holiday categories into categoricals - yes(=1) and no(=0)
    df.loc[df['StateHoliday'].isin(['a', 'b', 'c']), 'state_holiday'] = 1
    df.loc[~df['StateHoliday'].isin(['a', 'b', 'c']), 'state_holiday'] = 0
    df.drop(['StateHoliday'], axis=1, inplace=True)
    
    #create seasons from date
    df.loc[df['Date'].dt.month.isin([12, 1, 2]), 'winter'] = 1
    df.loc[~df['Date'].dt.month.isin([12, 1, 2]), 'winter'] = 0
    df.loc[df['Date'].dt.month.isin([3, 4, 5]), 'spring'] = 1
    df.loc[~df['Date'].dt.month.isin([3, 4, 5]), 'spring'] = 0
    df.loc[df['Date'].dt.month.isin([6, 7, 8]), 'summer'] = 1
    df.loc[~df['Date'].dt.month.isin([6, 7, 8]), 'summer'] = 0
    df.loc[df['Date'].dt.month.isin([9, 10, 11]), 'fall'] = 1
    df.loc[~df['Date'].dt.month.isin([9, 10, 11]), 'fall'] = 0
    
    #One hot encoding for store type, assortment
    cols = ['StoreType', 'Assortment']
    for col in cols:
        dummies = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,dummies],axis=1)
        df.drop(col, axis=1, inplace=True)
    
    #convert Competition month-year into datetime
    df['D'] = 1.0     #helper column for day to be added to date
    df['Competition_open_since'] = pd.to_datetime(df.CompetitionOpenSinceYear*10000+df.CompetitionOpenSinceMonth*100+df.D, format='%Y%m%d')
    
    #determine whether Competition was active at the point in time
    comp = df['Competition_open_since'] <= df['Date']
    df.loc[comp, 'competition_active'] = 1
    df.loc[~comp, 'competition_active'] = 0
       
    #convert Promo 2 week-year into datetime
    df['helper_date'] = df.Promo2SinceYear * 1000 + df.Promo2SinceWeek * 10 + 0
    df['Promo_since'] = pd.to_datetime(df['helper_date'], format='%Y%W%w')
    
    #determine whether Promo2 was active during the month
    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    promo_start_later = df['Promo_since'] > df['Date']

    for key, value in months.items():
        df.loc[(df['PromoInterval'].str.contains(key)) & (df['Date'].dt.month == value), 'Promo2_active'] = 1
    df.loc[promo_start_later, 'Promo2_active'] = 0
    df['Promo2_active'].fillna(0, inplace=True)
    
    #create date/datetime features
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear
    
    #delete irrelevant columns
    df.drop(['D',
            'helper_date',
            'CompetitionOpenSinceYear',
            'CompetitionOpenSinceMonth',
            'Competition_open_since',
            'Promo2SinceYear',
            'Promo2SinceWeek',
            'Promo2',
            'PromoInterval',
            'Promo_since',
            'DayOfWeek',
            ], axis=1, inplace=True)
    
    
    
    #### DELETE WHEN MATIAS IS DONE WITH DATA CLEANING###
    df = df.dropna()
    mask_0_sales = df['Sales'] != 0    
    df = df.loc[mask_0_sales, :]
    #df.set_index('Date', inplace=True)
    #### DELETE WHEN MATIAS IS DONE WITH DATA CLEANING###
    
    return df



def split_x_y(df, label=None):
    #### DELETE WHEN MATIAS IS DONE WITH DATA CLEANING###
    mask_0_sales = df['Sales'] != 0    
    df = df.loc[mask_0_sales, :]

    #### DELETE WHEN MATIAS IS DONE WITH DATA CLEANING###
    
    df.set_index('Date', inplace=True)
    
    X = df
    
    if label:
        X = df.drop(label, axis=1)
        y = df[label]
        
        return X, y

    return X


def scaler(X, mode='train', scaler=None):
    #scaler = StandardScaler()
    if mode == 'train':
        scaled_X = scaler.fit_transform(X)
    else:
        scaled_X = scaler.transform(X)
    
    return scaled_X, scaler



def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def model_fitting(X, y):
    model = xgb.XGBRegressor(
                            n_jobs=-1, 
                            n_estimators=1000, 
                            max_depth=6)
    model.fit(X, y)
    
    dump(model, './fitted_model.joblib') 
    fitted_model = load('./fitted_model.joblib')
    
    return model 
   

def predict(X_test, y_test, model):
    model = model
    preds = model.predict(X_test)
    
    return metric(preds, np.array(y_test)) 
    


def process_test(train_name='./data/clean_data.pkl', test_name='./test.csv'):
    #test data
    test_file = pd.read_csv(test_name)
    test_file.loc[:, 'Date'] = test_file.loc[:, 'Date'].astype('datetime64')
    test_file.drop('Open', axis=1, inplace=True)
    
    
    store_data = pd.read_csv('./store.csv')
    
    merge_file = test_file.merge(store_data, on = 'Store', how='left')
    
    test_feat = feature_engineering(merge_file)
    
    
    #train data
    train_file = pd.read_pickle(train_name)
    train_feat = feature_engineering(train_file)
    
    
    #split & scaling
    X_train, y_train = split_x_y(train_feat, label='Sales')
    X_test, y_test = split_x_y(test_feat, label='Sales')
    
    X_train_scaled, sclr = scaler(X_train, mode='train', scaler=StandardScaler())
    X_test_scaled, _ = scaler(X_test, mode='test', scaler=sclr)
    
    model = load('./fitted_model.joblib')
    
    print("Team3's RMSPE for the test set applying XGBoost is: {:.2f}%.\n\nOur hyperparameters: n_estimates = 1000,  max_depth=6\n\nData is cool! :D".format(predict(X_test_scaled, y_test, model)))
    
    return predict(X_test_scaled, y_test, model)
  


#EDA#
def train_val_split(df, date):
    #sort dataframe by Date and Store ID
    df.sort_values(by=['Date', 'Store'], ascending=True, inplace=True, ignore_index=True)
    
    #split dataset into features and target
    k = df[df['Date'] == date].index.max()
    data_train = df.loc[:k, :]
    data_val = df.loc[k+1:, :]
    
    assert data_train['Date'].max() < data_val['Date'].min()
    
    #returns train and validation datasets
    return data_train, data_val

def plot_predictions(X_train, y_train, X_test, y_test, predictions):
    Xy_train = pd.concat([X_train, y_train], axis=1)
    Xy_test = pd.concat([X_test, y_test], axis=1)
    test_start = min(Xy_test.index)
    test_end = max(Xy_test.index)
    
    Xy_test['Prediction'] = predictions
    df_all = pd.concat([Xy_train, Xy_cv, Xy_test], sort=False)
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))
    
    df_all.groupby('Date').sum()[['Sales','Prediction']].plot(ax=ax1)
    ax1.set_title('Forecast vs Actuals', fontsize= 20)
    
    df_all.groupby('Date').sum()[['Sales', 'Prediction']].plot(ax=ax2, style=['-','o'])
    ax2.set_xbound(lower=test_start, upper=test_end)
    ax2.set_title('Forecast vs Actuals - {} {} onwards'.format(min(Xy_test.index).strftime("%b"), min(Xy_test.index).year), fontsize= 20)
    
    fig.savefig('prediction.png')
