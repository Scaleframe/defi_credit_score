import os
import json
import lightgbm
import random
import pandas as pd
import numpy as np
import datetime
import time
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

    # relevant cutoff dates
    APR_15_2021 = time.mktime(datetime.datetime(2021,4,15).timetuple())
    JAN_1_2021 = time.mktime(datetime.datetime(2021,1,1).timetuple())

    # set this to True for the more aggressive test
    out_of_time_test = True

    # set this to enforce at least 3 months of future data, this eliminates
    # another possible "cheat"
    enforce_3months_future = True

    random.seed(1234)
    for u in users:
        evs = users[u]
        evs.sort(key = lambda x: x["timestamp"])

        if random.uniform(0,1) < train_frac: D = Dtrain
        else: D = Dtest

        for ev in evs:
            if ev["event_type"] != "borrow": continue

            feats = get_features_and_label(evs, ev["timestamp"])
            for fk in feats:
                if fk not in D: D[fk] = []
                D[fk].append(float(feats[fk]))

        

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

    # we're going to iterate over the keys, deleting one at a time
    # to check impact on AUC
    feat_keys= [x for x in Dtrain.keys() if x != "label"]
    
    # get the baseline ROC AUC score
    model = lightgbm.train(params, TR)
    preds = model.predict(df_te)
    orig_roc = roc_auc_score(target_te,preds)

    print(f"roc auc score {orig_roc}")

    # need an original copy of the training data since we're going
    # to be zeroing out columns
    df_tr_orig = df_tr.copy(deep=True)

    importance = {}

    for fk in feat_keys:
        df_tr_curr = df_tr_orig.copy(deep=True)
        df_tr_curr[fk] = df_tr_curr[fk]*0.0 # act as if we didn't have this feature
        TR = lightgbm.Dataset(df_tr_curr,label=target_tr)
        model = lightgbm.train(params, TR)
        preds = model.predict(df_te)
        curr_roc = roc_auc_score(target_te,preds)
        importance[fk] = orig_roc - curr_roc
        with open("/data/importance.json","wt") as f: 
            json.dump(importance, f, sort_keys=True,indent=2)
        print(fk, orig_roc - curr_roc)

