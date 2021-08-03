from io import RawIOBase
import os
import json


# This could be made more efficient without re-traversing the list
# comprehensions every time, but it's not a bottleneck for data at this volume

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

    # this is our training label, liquidation in the "near" future.
    # careful note: 1 means "credit_ok", which means *no* near term liquidation.
    near_liqs = [e for e in near_evs if e["event_type"] == "liquidation_call"]
    credit_ok = 1 if len(near_liqs) == 0 else 0

    # start assembling a feature map
    feats = {}
    feats["label"] = credit_ok

    
    # number and volume of past transactions by type
    types = "unknown deposit liquidation_call repay borrow".split()

    # use this to calculated a blended historical interest rate
    wsum_interest = 0.0 # rate * amount (numerator for weighted avg)
    wsum = 0.0 # sum of amounts (denominator for weighted avg)

    # track all the distinct pools and reservers and symbols
    pools = {}
    reserves = {}
    symbols = {}

    for typ in types:
        # for each event type get sums, nums (counts), and averages
        num_past_events = 0
        sum_past_events = 0

        for e in past_evs:
            # use for tracking distinct pool and reserve data
            if "pool_id" in e: pools[e["pool_id"]] = True
            if "reserve_id" in e: reserves[e["reserve_id"]] = True
            if "reserve_symbol" in e: symbols[e["reserve_symbol"]] = True

            # break early if past event doesnt match types we are aggregating
            if e["event_type"] != typ: continue

            # add to count of past events
            num_past_events += 1.0

            # handle adding to sum of past events (ones that contain an "amount" value)
            for amnt in "amount amountAfterFee collateralAmount".split():
                if amnt in e:
                    sum_past_events += int(e[amnt])
                    break

            # handle interest (check borrows only for now)
            if typ != "borrow": continue

            # change ray units to decimals, ray is 27 decimals of precision:
                # https://docs.aave.com/developers/v/1.0/developing-on-aave/important-considerations#ray-math

            rate = float(e["borrowRate"])

            # sum numerator and denominator for calculating weighted avg
            wsum_interest += rate * float(e["amount"])
            wsum += float(e["amount"])

        # find blended interest rate with sums from past events
        weighted_interest = wsum_interest / max(1.0, wsum)


        # update averages, sums, nums, and weighted interest to feature map
        avg_past_events = sum_past_events/max(1.0, float(num_past_events))
        feats[typ + "_num"] = num_past_events
        if typ != "unknown":
            feats[typ + "_sum"] = sum_past_events
            feats[typ + "_avg"] = avg_past_events

        if typ == "borrow":
            feats["weighted_interest"] = weighted_interest

    # update unique values for pools and reserves.
    feats["num_pools"] = len(pools)
    feats["num_reserves"] = len(reserves)
    feats["num_symbols"] = len(symbols)

    return feats

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
    # build feature dict for all events for a given user in the user mapping
    for usr in users:
        print(f"extracting features for user:{usr}...")
        evs = users[usr]
        evs.sort(key = lambda x: x["timestamp"])

        # get features and labels from borrow events
        for ev in evs:
            if ev["event_type"] != "borrow": continue
            feats = get_features_and_label(evs, ev["timestamp"])

    print(f"\nfeatures extracted for all users successfully.\n")
    print(f"\nfeature column example:\n{feats}")


    # feature columns:
        # {'label': 1,
        #  'unknown_num': 0,
        #  'deposit_num': 1.0,
        #  'deposit_sum': 6427407312330356777,
        #  'deposit_avg': 6.427407312330357e+18,
        #  'liquidation_call_num': 0,
        #  'liquidation_call_sum': 0,
        #  'liquidation_call_avg': 0.0,
        #  'repay_num': 0,
        #  'repay_sum': 0,
        #  'repay_avg': 0.0,
        #  'borrow_num': 0,
        #  'borrow_sum': 0,
        #  'borrow_avg': 0.0,
        #  'weighted_interest': 0.0,
        #  'num_pools': 1,
        #  'num_reserves': 1,
        #  'num_symbols': 1}
