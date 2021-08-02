import sys
import os
import json
import requests
import random


def _get_event_type(out_dict):
    """
        get the event type of an event based on keys present.
    """
    keys = set(out_dict.keys())

    if "amountAfterFee" in keys:
        return "repay"

    if "liquidator" in keys:
        return "liquidation_call"

    if "borrowRate" in keys:
        return "borrow"
    
    if "amount" in keys:
        return "deposit"

    return "unknown"



def process_response(json_data, depth=2, single_values=False):
    """
        process json response into a format that can be used for modeling.

        original:
            "data": {
                "userTransactions": [
                {
                    "amount": ,
                    "id": ,
                    "pool": {
                        "id": ,
                    }
                    "reserve": {
                        "id": ,
                        "symbol": ,
                    },
                    "timestamp":,
                    "user": {
                        "id":
                    }
                }
            }
        }
        processed:
        {
            "amount": ,
            "pool_id": ,
            "pool_lendingPool": ,
            "reserve_id": ,
            "reserve_symbol": ,
            "timestamp": ,
            "user_id": ,
            "txn_id": ,
            "event_type":
        }
    """
    output = []

    # list of dicts
    user_transactions = json_data["data"]["userTransactions"]

    for data in user_transactions: 

        # flatten / de nest the data, two levels depth is sufficient here.
        denested_data = _denest_data(data, depth, single_values=single_values)

        # change id to transaction id
        txn_id = denested_data.pop("id")
        denested_data["txn_id"] = txn_id

        # return event type string (borrow, repay, liquidation)
        event_type = _get_event_type(denested_data)
    
        # add the event type to the output dict
        denested_data.update(event_type=event_type)
  
        output.append(denested_data)

    return output


def _denest_data(data, target_depth, traversed_depth=0, initial_key=None, single_values=False):
    
    """
        Traverse the nested data and flatten. Bringing single values one level up by joining the lower level key to the keyname one level higher.
        Example:
            pool: {
                id: "value"
            }
        becomes:
            {
                pool_id: "value"
            }
    """

    # lets us recursively call denest until we get to the depth we want.
    if traversed_depth >= target_depth:
        return "...Truncated..." if single_values else data

    traversed_depth += 1

    if isinstance(data, dict):
        output = {}
        for key, value in data.items():
            # update the lower level key name to the higher level key name
            key_name = f"{initial_key}_{key}" if initial_key else key

            # continue to denest recursively if we have any lists or dicts
            if isinstance(value, (list, dict)):  # [{}, {}, ...] # [{}, {}, ...]
                ret = _denest_data(value, target_depth, traversed_depth, initial_key=key_name, single_values=single_values)
            else:
                ret = value
            # after recusion we return all values found iterating at lower
                                                    # levels to the output
            if isinstance(ret, dict):
                output.update(ret)
            else:
                output[key_name] = ret
    # handle denesting lists passed to the function
    elif isinstance(data, list):
        output = []
        for item in data:
            ret = _denest_data(item, target_depth, traversed_depth, single_values=single_values)   
            output.append(ret)         
    # handle single values
    else:
        output = data
        
    return output


def get_query(timestamp):
    """
        get a query string with a variable timestamp (for fetching multple queries)
    """
    return (
r"""query Query($userTransactionsOrderBy: UserTransaction_orderBy) {userTransactions(first: 1000, orderBy: timestamp, where: {timestamp_lt:""" + str(timestamp) + r""" }, orderDirection: desc) {
    id
    timestamp
    user {
      id
    }
    ... on Borrow {
      id
      reserve {
        id
        symbol
      }
      amount
      borrowRate
      borrowRateMode
      accruedBorrowInterest
      pool {
        id
        lendingPool
      }
    }
    ... on Repay {
      id
      pool {
        id
        lendingPool
      }
      amountAfterFee
      fee
      timestamp
      reserve {
        id
        symbol
      }
    }
    ... on LiquidationCall {
      id
      timestamp
      principalAmount
      liquidator
      pool {
        id
        lendingPool
      }
      collateralAmount
      collateralReserve {
        id
        underlyingAsset
      }
      principalReserve {
        id
        underlyingAsset
      }
    }
    ... on Deposit {
      id
      timestamp
      amount
      pool {
        id
        lendingPool
      }
      reserve {
        id
        symbol
    }
  }
}}
"""
    ).rstrip("\n")

def graphql_query(query):
    """
        Pass query to graphql endpoint and retrieve a json object.
    """
    url = 'https://api.thegraph.com/subgraphs/name/aave/protocol-multy-raw'

    headers = {'content-type': 'application/json'}

    r = requests.post(url, json={'query': query.replace('\n', '')}, headers=headers)
    processed_data = process_response(r.json())
    return processed_data
    

def grab_all_events():
    """
        This works around the 1000 record query limit with  the aave graphql api.

        We ask for records that have a timestamp earlier than the previous 
        earliest timestamp, and work down from there to get all the records.
    """

    final_output = []

    oldest_timestamp = 1578505854
    current_timestamp = 1911111111
    while current_timestamp > oldest_timestamp:

        # start with a recent timestamp and grab the newest records
        query = get_query(current_timestamp)


        # call the graphql api with the desired query
        event_batch = graphql_query(query)

        final_output.extend(event_batch)
        
        # if we have data, append to the event_list
        if len(event_batch) > 0:
            print(f"appending data to event_list")

        # if we've run out of data, break from the loop
        else:
            print("no data left, breaking fetch cycle")
            break

        # use the earliest timestamp as the start of the next batch
        
        # get earliest timestamp
        timestamps = [i["timestamp"] for i in event_batch]
        print(f"timestamps: {timestamps}")

        earliest_batch_timestamp  = min(timestamps)
        print(f"smallest timestamp: {min(timestamps)}")
        current_timestamp = earliest_batch_timestamp
        print(f"new earliest timestamp: {current_timestamp}")
    
    return final_output

 
def get_user_mapping(events):
    """
        get mapping of events to user key
        {
            "user_id": [
                {},{},{}
            ],
            "user_id": [
                {},{},{}
            ]
        }
    """
    user_mapping = {}

    for event in events:
        user_id = event["user_id"]
        # current_values = events_mapping.get(event_type, [])
        # current_values.append(event)
        user_mapping.setdefault(user_id, []).append(event)

    return user_mapping


def get_test_data_sample(events, num_samples=50000):
    """
        create test data and save to disk
    """

    random.seed(1234)
    test_data = random.sample(events, num_samples) if len(events) > num_samples else print("dataset too small, use the whole file or decrease number of samples")
    
    with open("./data/test_data.json", "wt") as f:
        json.dump(test_data, f, indent=2)
    print(f"test data created")

    return test_data

def get_test_data_mapping(test_data):
    """
        get mapping from sample of event logs
    """
    test_data_mapping = get_user_mapping(test_data)
    return test_data_mapping

def run_full_fetch():
    """
        fetch data from api and save to disk.
    """
    print(f"Retrieving all events from graphql api...")
    all_events = grab_all_events()
    
    print(f"events retrieved, saving to disk")
    with open("./data/all_events.json", "wt") as f:
        json.dump(all_events, f, indent=2)
    return True

if __name__ == "__main__":

    # check data directory
    print("checking for data directory")
    data_dir_name = '/data/'
    data_dir = os.getcwd()+ "/data/"
    if os.path.isdir(data_dir) is False:
        print(f"creating data directory")
        os.makedirs('./data')
    else:
        print("data directory found")

    # if the -fetch flag is set, fetch the data from the api
    if sys.argv[1] == '--fetch':
        run_full_fetch()

    # if data is already present on disk, skip running a full fetch.
    print("checking for event data on disk ...")
    data_file = os.getcwd() + '/data/all_events.json' # get file location
    if os.path.isfile(data_file):
        print("\"all_events.json\" found, loading from disk.")
        with open('./data/all_events.json') as f:
            all_events = json.load(f)
    else:
        # if not data on disk, fetch the data from the api
        run_full_fetch()
        

    # create mapping of user transasction from event logs
    user_mapping = get_user_mapping(all_events)
    
    # save mapping to disk      
    print("saving user_mapping ...")
    with open("./data/all_user_mapping.json", "wt") as f:
        json.dump(user_mapping, f, indent=2)    
    print("success")


    # Uncomment this block to create a smaller sample of test data and save to disk. Not neccesary at this data volume but may change as usage of Aave's service grows. 
    
    # all_events = []
    # if not len(all_events):
    #     with open('./data/all_events.json') as f:
    #         all_events = json.load(f)
    # test_data = get_test_data_sample(all_events)
    # test_data_mapping = get_test_data_mapping(test_data)
    # print(test_data_mapping)
    # with open('./data/test_data_mapping.json', "wt") as f:
    #     json.dump(test_data_mapping, f, indent=2)

   