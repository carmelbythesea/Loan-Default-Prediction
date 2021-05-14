def prepare_data(data, path, name, max_sample):

    from sklearn import preprocessing
    import category_encoders as ce
    import pandas as pd
    print("start of prepare_data")

    cat_final = ['professionid', 'birthplace', 'residencezipcode', \
    'companyzipcode', 'legalzipcode', 'education', 'maritalstatus']
    quant = ['numofdependence', \
         'monthlyfixedincome', 'monthlyvariableincome', 'spouseincome', \
         'avg_income', 'std_income', 'avg_income_cnt', 'avg_income_nation', 'std_income_nation',
       'avg_income_nation_cnt', 'avg_income_area', 'std_icnome_area',
       'avg_income_area_cnt', 'avg_sale_house_price_5000',
       'std_sale_house_price_5000', 'sale_house_cnt_5000',
       'avg_sale_apartment_price_5000', 'std_sale_apartment_price_5000',
       'sale_apartment_cnt_5000', 'avg_rent_house_price_5000',
       'std_rent_house_price_5000', 'rent_house_cnt_5000',
       'avg_rent_apartment_price_5000', 'std_rent_apartment_price_5000',
       'rent_apartment_cnt_5000', 'avg_sale_house_price_10000',
       'std_sale_house_price_10000', 'sale_house_cnt_10000',
       'avg_sale_apartment_price_10000', 'std_sale_apartment_price_10000',
       'sale_apartment_cnt_10000', 'avg_rent_house_price_10000',
       'std_rent_house_price_10000', 'rent_house_cnt_10000',
       'avg_rent_apartment_price_10000', 'std_rent_apartment_price_10000',
       'rent_apartment_cnt_10000', 'previous_converted']
    # resample_df = pd.read_csv(path+name)
    resample_df = data
    final_df = resample_df[cat_final]
    final_df['index'] = resample_df.index
    # label encoder on ordinal features
    le = preprocessing.LabelEncoder()
    final_df['professionid'] = le.fit_transform(resample_df.professionid)
    final_df['education'] = le.fit_transform(resample_df.education)

    non_ordinal = set(cat_final)^set(['professionid', 'education'])
    # Hashing Encoding for large scale categorical data
    HE = ce.HashingEncoder(cols = non_ordinal, return_df=True, max_sample = max_sample)
    # encode the categorical variables
    data = HE.fit_transform(final_df)

    #categorical features fillna with mode
    data = data.apply(lambda x: x.fillna(x.mode()), axis=0)
    #quantitative features fillna with mean
    quant_data = resample_df[quant].apply(lambda x: x.fillna(x.mean()), axis=0)

    X = quant_data.reset_index().merge(data, on = 'index').drop('index', axis = 1)
    y = resample_df['defaulted']

    print("Shape of X, y: ", X.shape, y.shape)
    return X, y
