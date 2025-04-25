import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(test_size = 0.2, random_state = 42): #non GPU version lol
    data = load_breast_cancer()
    x, y = data.data, data.target

    scalar = StandardScaler()
    x = scalar.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                        test_size=test_size, 
                                        random_state= random_state, 
                                        stratify=y)
    
    return {
        "x_train": x_train.astype(np.float32),
        "y_train": y_train.reshape(-1, 1).astype(np.float32),
        "x_test": x_test.astype(np.float32),
        "y_test": y_test.reshape(-1, 1).astype(np.float32), 
        "feature_names": data.feature_names,
        "target_names": data.target_names
        }