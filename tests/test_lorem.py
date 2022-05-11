import pytest

import sys
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from lore_explainer.util import record2str, neuclidean
from lore_explainer.datamanager import prepare_adult_dataset, prepare_dataset

def test_datapreparation():

    df, class_name = prepare_adult_dataset('adult.csv')
    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(
        df, class_name)

    assert df.shape[0] > 1, "Loaded table shoudl contain at least one record"