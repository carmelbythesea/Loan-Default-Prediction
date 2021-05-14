def train_data(X, y, test_size_pct, max_depth, min_samples_leaf, cv_split):

    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    import timeit
    import pandas as pd
    import numpy as np

    print("start of train_data")
    start = timeit.default_timer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
        test_size = test_size_pct, random_state=1)

    # start to train X
    output = {}
    clf = DecisionTreeClassifier(max_depth = max_depth,\
        min_samples_leaf = min_samples_leaf)
    clf.fit(X_train, y_train)
    stop = timeit.default_timer()
    print("training time: ", stop - start)
    # accuracy of training prediction
    output['clf_train_pred'] = sum(clf.predict(X_train) == y_train)/len(X_train)

    # accuracy of test prediction
    start = timeit.default_timer()
    pred = clf.predict(X_test)
    stop = timeit.default_timer()
    print("testing time: ", stop - start)

    output['clf_test_pred'] = sum(pred == y_test)/len(X_test)

    # roc_auc score of 5 split train samples
    output['mean_train_cvs'] =np.mean(cross_val_score(estimator = clf, \
        y = y_train, X = X_train, cv=cv_split, scoring='roc_auc'))
    # roc_auc_score of 5 split test samples
    output['mean_test_cvs'] =np.mean(cross_val_score(estimator = clf, y = y_test, \
                                                X = X_test, cv=cv_split, scoring='roc_auc'))

    print(output)
    return clf
