
# Defi Credit Score: 

This repo analyzes user behavior on blockchain based defi networks. It models credit risk using only "on chain" data, as opposed to private personal information that would be supplied for traditional credit. 

This solution focuses on predicting a given user's probability of their loan being liquidated 3 months into the future, given their previous 6 months interactions with the lending pool. 

This achieves an _ROC AUC score_ of .79 on Aave's smart contract transactions from 2020 and 2021. 

## Reproducing These Results

Install Dependencies: In a new python environment, run:

`pip install -r requirements.txt`

To run the model, you will need to fetch the Aave smart contract transactions and save the data. Run:

`python graphql-fetcher.py`

If you have data on disk the fetcher will load from there automatically. To force the fetcher to run a full fetch, add the --fetch option:

`python graphql-fetcher.py --fetch`

Build Features, Test Models, Find Feature Importance:

run the numbered python scripts in your favorite terminal / notebook to create features, analyze the data, and view predictions:

`python 01-feature-engineering.py`

`python 02-credit-scoring.py`

`python 03-credit-scoring-aggressive-randomize.py`

`python 04-feature-importance.py`

## Solution Flow Notebook:

You can also closely follow the solution flow of this repo and interact with the code yourself by using the `01-data-and-solution-flow-notebook.ipynb`. 

You'll have to run `graphql-fetcher.py` either in a notebook or your favorite terminal to get your hands on the data that the notebook and scripts use. 

## The Problem: 

Currently in Defi, credit is heavily collteralized, meaning users can borrow a currency given they put down an equal amount or more as collateral. This is not only capital inefficient, but also misses the promise of defi; to provide widespread access to financing, regardless of a users background or access to traditional capital. 

## The Objective:

The goal of this project is to give users a "defi credit score" that allow lenders to extend further credit while minimizing their exposure to risk. 

This will also provide users of lending services access to credit without having to reveal personal information about themselves, which aligns more closely with the spirit of crypto and blockchain technology. 

## The Data: 

Aave's data is made available through a GraphQL api:
https://docs.aave.com/developers/v/1.0/integrating-aave/using-graphql

A good way to visualise this data is to view the schema using a client service like Apollo: 
https://studio.apollographql.com/sandbox/explorer?endpoint=https://api.thegraph.com/subgraphs/name/aave/protocol-multy-raw


## Data Pipeline:

The data captures users interaction with Aave's smart contracts and servives. The graph based nature of the api is good for learning many things about a single data point, but presents challenges for data preparation and ml modeling. Namely:

- data is heavily nested / circular
- the api client is limited to 1000 records per call
- api records are not labled explicity by event type

`graphql_fetcher.py` solves these problems by providing methods to extract, process, and sort event log data from the graphql api. 

Inlcuded are functions to: 

- extract all the avaialable records, 
- label events by their type, 
- de nest / flatten data to the desired level / dimenstionality, and  
- create sample files for downstream tests.  


## Model Formulation: 

Here a users credit score is modeled as a supervised learning problem, with the target label being the probability of a liquidation event. In the nascent defi space, this is a close approximation for a default. 

This looks at every time an account borrows from a pool, and assign a just-in-time credit 
score. The score will model the probability of a liquidation within 3 months, using the previous 6 months transactions data. This gives us a way to give a credit score for any account at any point in time. The 3-month and 6-month time windows can be adapted depending on business needs.


## Feature Extraction

This uses aggregated data from the previous 6 month period as predictive features. It looks at the count of each kind of transaction, and the total value of transactions that have monetary value (e.g. total value of repayments in the previous 6 months). 

This also calculates a weighted average of interest rate on all previous borrow events (weighted by monetary value), as well as the number of pools, reserves, and symbols they have interacted with. In the future this could also look to see if specific pools or reserves are predictive, as opposed to just a count.

## Modeling Tool

This uses Gradient Boosting, a popular modeling tool for financial data. Specifically I use Python LightGBM, a library for implmementing gradient boosting. GBM is a generalization of decision trees, instead of one decision tree you have many (in this case 200) and they all contribute a small part to the decision.


## Performance metric

Here I use ROC AUC as the performance metric because it is the dominant metric in credit risk scoring. It captures a model's ability to do relative risk ranking, as opposed to hard yes/no predictions of repay vs default.

## Testing Framework

This uses an out-of-time, out-of-population test. It trains on one population (66% of users) using all their data up to Jan 1 2021, and tests on a separate population (33% of users) using their transactions since Jan 1 2021. This forces the model testing to deal with both unseen borrowers and an unseen time period.

## Feature importance

There are many ways to do this but let's use a cheap, low-tech, and easy to understand method. Build a model using all the features, then build sub-models hiding one of the features. The loss of performance gives a measure of feature importance. In these experiments, the number of historical borrow events and the total value of all recent repayments seem to be the top predictors.

## Results

The model achieved a ROC AUC score of .79 on the test set.    

## In This Repo:

00: graphql_fetcher.py:
- extract and process blockchain transaction data 

01: feature engineering: 
- build features for ml model from processed data

02: credit modeling, first pass:
- first pass at predicting borrow liquidation

03: credit modeling, second pass:
- stress testing the model and further improvements

04: feature importance:
- remove features to measure impact on model performance
 