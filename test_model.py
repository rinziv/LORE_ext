import pickle
import numpy as np

if __name__ == '__main__':
    model_file = 'models/adult_rf_lore.sav'
    
    x = np.array([38,28887,7,0,0,
            50,0,0,0,1,0,0,
            0,0,0,1,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,1,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,
            1,0,0,1,0,0,0,0,0,0,0,0,
            0,1,0,1,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,1,0,0])



    # Export the model to a file
    infile = open(model_file, 'rb')
    lore_explainer = pickle.load(infile)
    infile.close()

    # Printing a sample tuple
    print(x)
    exp = lore_explainer.explain_instance(x, samples=300, use_weights=True)
    str = str(exp)
    print(str)