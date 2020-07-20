def merge_files(left_df=train, right_df=store):
    df = left_df.merge(right_df, on='Store', how='left')
    return df
