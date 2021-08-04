import os
import json
import lightgbm
import random
import pandas as pd
import numpy as np
import importlib
from sklearn.metrics import roc_auc_score

# same as "from 01-feature-engineering import get_features_and_label"
get_features_and_label = importlib.import_module('01-feature-engineering').get_features_and_label


if __name__ == "__main__":

    print("checking for user mapping on disk ...")
    data_file = os.getcwd() + '/data/all_user_mapping.json'
    if os.path.isfile(data_file):
        print("\"all_user_mapping.json\" found, loading from disk.")
        with open("./data/all_user_mapping.json") as f:
            users = json.load(f)
    elif os.path.isfile(data_file) == False:
        raise FileNotFoundError("User mapping file \"all_user_mapping.json\" not present in data directory. Try running `graphql_fetcher.py` to retrieve and store the data.")

    print("data successfully loaded from disk.")

    print("Running credit score predictions...")
    Dtrain = {}
    Dtest = {}
    train_frac = 0.66 # 2/3 of data used to train

    random.seed(1234)
    for usr in users:
        evs = users[usr]
        evs.sort(key = lambda x: x["timestamp"])

        # train / test split, randomly assign users to either train or test groups 
                                # D points to Dtrain or Dtest depending on split
        if random.uniform(0,1) < train_frac:
            D = Dtrain
        else:
            D = Dtest

        for ev in evs:
            if ev["event_type"] != "borrow": continue

            # append lists of features to each key column from every event
             # D {
                # label: [{event1} {event2}],
                # feat1: [{event1}, {event1}]
             # }
            feats = get_features_and_label(evs, ev["timestamp"])
            for fk in feats:
                if fk not in D: D[fk] = [] # set default as a list
                D[fk].append(float(feats[fk])) # update dict with list of values from each event

    # create dataframes from the built up dicts
    df_tr = pd.DataFrame.from_dict(Dtrain)
    df_te = pd.DataFrame.from_dict(Dtest)

    target_tr = df_tr.label.values
    target_te = df_te.label.values

    df_tr.drop(["label"],inplace=True,axis=1)
    df_te.drop(["label"],inplace=True,axis=1)

    TR = lightgbm.Dataset(df_tr,label=target_tr)
    TE = lightgbm.Dataset(df_te,label=target_te)

    MD = 3 # max depth for the treess
    NE = 200 # number of trees in the gradient boosting model
    TD = 2.0 # total distance searched by the gradient boosting iteration

    params = {
        "n_estimators" : NE,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': int(2**MD),
        'learning_rate': TD/NE,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'verbose': 0,
        'max_depth': MD
    }

    model = lightgbm.train(params, TR)
    preds = model.predict(df_te)

    print(roc_auc_score(target_te,preds))

