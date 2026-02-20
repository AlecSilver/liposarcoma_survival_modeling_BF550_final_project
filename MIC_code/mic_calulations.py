

import numpy as np
import pandas as pd
from minepy import pstats, cstats


np.random.seed(42)

X = pd.read_csv('/app/data/combined_expression_data_scaled.csv').as_matrix()
y = pd.read_csv('/app/data/combined_metadata.csv')
y = y[['metastasis_status', 'time']]
#convert status to int 
y['metastasis_status'] = y['metastasis_status'].astype(int)
y = y.as_matrix().T

print("X shape:", X.shape)
print("y shape:", y.shape)
# compute statistics between each pair of samples in X and Y
mic_c, tic_c =  cstats(X, y, alpha=9, c=5, est="mic_e")


# save MIC results
np.savetxt('/app/data/mic/mic_c.csv', mic_c, delimiter=',')