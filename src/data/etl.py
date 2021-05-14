def clean_data(path, file_name, N_samples, rs):
    import pandas as pd
    import numpy as np
    from sklearn.utils import resample

    print("start of clean_data")
    df = pd.read_csv(path +"/"+ file_name)
    # create the target column
    # If maxoverduedays>90, we say that the borrower has defaulted on the loan.
    df['defaulted'] = df['MaxOverDueDays'] > 90

    #Seperate each target class into 2 dataframes
    not_default = df[df['defaulted'] == 0]
    default = df[df['defaulted'] == 1]

    #Resample dataframe
    resample_default = resample(default,
                           replace = False,
                           n_samples = N_samples,
                           random_state = rs)

    resample_nondefault = resample(not_default,
                           replace = False,
                           n_samples = N_samples,
                           random_state = rs)

    resample_df = pd.concat([resample_default, resample_nondefault])

    # previous: 1 means the person has record on loaning
    resample_df['previous_converted']= resample_df.previous.apply(lambda x: 0 if pd.isnull(x) else 1)
    resample_df.to_csv(path+"/"+"resample_df.csv")
    return resample_df
