import sys
from pathlib import Path
import os
import mlflow

MLFLOW_TRACKING_URI=os.environ['MLFLOW_TRACKING_URI']
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
pretrained_model="squeezebert/squeezebert-uncased"

mapping = {
    0:"admiration",
    1:"amusement",
    2:"anger",
    3:"annoyance",
    4:"approval",
    5:"caring",
    6:"confusion",
    7:"curiosity",
    8:"desire",
    9:"disappointment",
    10:"disapproval",
    11:"disgust",
    12:"embarrassment",
    13:"excitement",
    14:"fear",
    15:"gratitude",
    16:"grief",
    17:"joy",
    18:"love",
    19:"nervousness",
    20:"optimism",
    21:"pride",
    22:"realization",
    23:"relief",
    24:"remorse",
    25:"sadness",
    26:"surprise",
    27:"neutral",
}