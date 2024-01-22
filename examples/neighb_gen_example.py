from lore_explainer.neighgen import RandomGenerator

from lore_explainer.datamanager import prepare_adult_dataset, prepare_dataset
from lore_explainer.util import calculate_feature_values
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def run():
    df, class_name = prepare_adult_dataset('../adult.csv')
    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(
        df, class_name)

    print(df.shape)

    # Prepare data for learning method

    test_size = 0.30
    random_state = 0

    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_name].values,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=df[class_name].values)

    _, K, _, _ = train_test_split(rdf[real_feature_names].values, rdf[class_name].values,
                                  test_size=test_size,
                                  random_state=random_state,
                                  stratify=df[class_name].values)

    # Train a random forest model

    bb = RandomForestClassifier(n_estimators=20, random_state=random_state)
    # bb = MLPClassifier(random_state=random_state)
    bb.fit(X_train, Y_train)

    # bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
    # numeric_columns_index, ocr = 0.1

    print(K[0])

    numeric_columns_index = [i for i, c in enumerate(feature_names) if c in numeric_columns]
    nbr_features = len(feature_names)
    nbr_real_features = K.shape[1]
    feat_values = calculate_feature_values(K, numeric_columns_index, categorical_use_prob=False, continuous_fun_estimation=False)
    print(feat_values)

    neighgen = RandomGenerator(bb.predict, feat_values, features_map, nbr_features,
                               nbr_real_features, numeric_columns_index)

    N = neighgen.generate(X_train[10], 3)
    print(N)

if __name__ == '__main__':
    run()