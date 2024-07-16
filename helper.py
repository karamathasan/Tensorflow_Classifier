import pandas as pd
import numpy as np

def dataSplit(predictor_data: list, effector_data: pd.DataFrame, percentageTraining = 0.6):
    assert(len(predictor_data) == len(effector_data))
    rows = len(predictor_data)
    # shuffle(predictor_data, effector_data, shuffleProportion)
    training_predictor = predictor_data[0:int(rows * percentageTraining)]
    training_effector = effector_data.iloc[0:int(rows * percentageTraining)]
    testing_predictor = predictor_data[int(rows * percentageTraining): rows]
    testing_effector = effector_data.iloc[int(rows * percentageTraining): rows]
    return training_predictor, training_effector, testing_predictor, testing_effector

def shuffle(predictor_data:list, effector_data: pd.DataFrame, shuffleProportion = 0.5):
    assert len(predictor_data) == len(effector_data)
    rows = len(predictor_data)
    shuffleiterations = int(rows * shuffleProportion)
    for i in range(shuffleiterations):
        u = np.random.randint(0,len(predictor_data))
        v = np.random.randint(0,len(predictor_data))

        temp = predictor_data[u]
        predictor_data[u] = predictor_data[v]
        predictor_data[v] = temp
        
        temp = effector_data.iloc[u]
        effector_data.iloc[u] = effector_data.iloc[v]
        effector_data.iloc[v] = temp