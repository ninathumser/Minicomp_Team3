def merge_files(left_df=train, right_df=store):
    df = left_df.merge(right_df, on='Store', how='left')
    return df

def train_val_split():
    #read in pickle
    file_path = '../data/clean_data.pkl'
    df = pd.read_pickle(file_path)
    
    #sort dataframe by Store ID
    df.sort_values(by=['datetime', 'Store'], inplace=True, ignore_index=True)
    
    #split dataset into features and target
    #k = int(df.shape[0] * relative_train)
    k = df[df['datetime'] == '2014-03-30'].index.max()
    data_train = df.loc[:k, :]
    data_val = df.loc[k+1:, :]
    
    assert data_train['datetime'].max() < data_val['datetime'].min()
    
    #returns train and validation datasets
    return data_train, data_val
