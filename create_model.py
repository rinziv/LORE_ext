import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lore.datamanager import prepare_dataset
from lore.lorem import LOREM

if __name__ == '__main__':
    source_file = 'datasets/adult.csv'
    model_file = 'models/adult_rf_lore.sav'
    class_field = 'class'

    # Load and transform dataset and select one row to classify and explain
    df = pd.read_csv(source_file, skipinitialspace=True, na_values='?', keep_default_na=True)
    df, feature_names, class_values, numeric_columns, \
    rdf, real_feature_names, features_map = prepare_dataset(df, class_field)

    # Learn a model from the data
    test_size = 0.3
    random_state = 0
    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_field].values,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=df[class_field].values)

    bb = RandomForestClassifier(n_estimators=20, random_state=random_state)
    bb.fit(X_train, Y_train)

    lore_explainer = LOREM(rdf[real_feature_names].values, bb.predict, feature_names, class_field, class_values,
                           numeric_columns, features_map, neigh_type='geneticp', categorical_use_prob=True,
                           continuous_fun_estimation=False, size=300, ocr=0.1, random_state=random_state, ngen=10,
                           bb_predict_proba=bb.predict_proba, verbose=False)

    # Export the model to a file
    outfile = open(model_file, 'wb')
    pickle.dump(lore_explainer, outfile)
    outfile.close()