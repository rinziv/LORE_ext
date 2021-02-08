import pickle
import numpy as np

def explain_tuple(tuple):
    model_file = 'models/adult_rf_lore.sav'
    # Export the model to a file
    infile = open(model_file, 'rb')
    lore_explainer = pickle.load(infile)
    infile.close()

    # Printing a sample tuple
    print(tuple)
    exp = lore_explainer.explain_instance(np.array(tuple), samples=300, use_weights=True)
    return str(exp)