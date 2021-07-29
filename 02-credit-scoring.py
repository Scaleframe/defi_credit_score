import json
import lightgbm
import random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

with open('./data/all_user_mapping.json') as f:
    users = json.load(f)

def get_features_and_label (evs, timestamp):
    
    # split events into ones before and after this timestamp (exclude ==)
    prev_evs = [e for e in evs if e["timestamp"] < timestamp]
    fut_evs = [e for e in evs if e["timestamp"] > timestamp]

    # restrict past events to trailing 180 days (~6 months)
    past_window = 180*24*60*60 # 180 days in seconds
    past_evs = [e for e in prev_evs if e["timestamp"] >= timestamp - past_window]

    # from future events, get the ones in the next 90 days from timestamp
    fut_window = 90*24*60*60 # 90 days in seconds
    near_evs = [e for e in fut_evs if e["timestamp"] <= timestamp + fut_window]

    # this is our training label, liquidation in the "near" future
    near_liqs = [e for e in near_evs if e["event_type"] == "liquidation_call"]
    credit_ok = 1 if len(near_liqs) == 0 else 0

    # start assembling a feature map
    feats = {}
    feats["label"] = credit_ok


    # number and volume of past transactions by type
    types = "unknown deposit liquidation_call repay borrow".split()

    # we'll use this to calculated a blended historical interest rate
    wsum_interest = 0.0
    wsum = 0.0

    # we'll track all the distinct pools and reservers and symbols
    pools = {}
    reserves = {}
    symbols = {}

    for typ in types:
        num_past_events = 0
        sum_past_events = 0

    for e in past_evs:
        if "pool_id" in e: pools[e["pool_id"]] = True
        if "reserve_id" in e: reserves[e["reserve_id"]] = True
        if "reserve_symbol" in e: symbols[e["reserve_symbol"]] = True

        if e["event_type"] != typ: continue
        num_past_events += 1.0

        # the key for the amount
        for amnt in "amount amountAfterFee collateralAmount".split():
            if amnt in e:
                sum_past_events += int(e[amnt])
                break

        # handle interest
        if typ != "borrow": continue

        # take the first 8 digits of the interest rate
        assert len(e["borrowRate"]) <= 30
        rate = e["borrowRate"].rjust(30,"0")
        rate = int(rate[:8])

        wsum_interest += rate * float(e["amount"])
        wsum += float(e["amount"])

    weighted_interest = wsum_interest / max(1.0, wsum)


    # transaction features associated with this kind of event
    avg_past_events = sum_past_events/max(1.0, float(num_past_events))
    feats[typ + "_num"] = num_past_events
    if typ != "unknown":
        feats[typ + "_sum"] = sum_past_events
        feats[typ + "_avg"] = avg_past_events

    if typ == "borrow":
        feats["weighted_interest"] = weighted_interest

    feats["num_pools"] = len(pools)
    feats["num_reserves"] = len(reserves)
    feats["num_symbols"] = len(symbols)

    return feats

Dtrain = {}
Dtest = {}
train_frac = 0.66 # 2/3 of data used to train

random.seed(1234)
for u in users:
    evs = users[u]
    evs.sort(key = lambda x: x["timestamp"])

    for ev in evs:
        if ev["event_type"] != "borrow": continue

        feats = get_features_and_label(evs, ev["timestamp"])
        for fk in feats:
            if fk not in D: D[fk] = []
            D[fk].append(float(feats[fk]))

    if random.uniform(0,1) < train_frac: D = Dtrain
    else: D = Dtest

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

