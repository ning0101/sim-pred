import numpy as np
import tensorflow as tf
import glob
import pandas as pd


# data_path = 'ml-100k/ml-100k-prediction-result-c/'
data_path = 'ml-1m/ml-1m-prediction-result-c/'

rating = []
for j in range(35):
    df = pd.read_csv(data_path + 'item_adcos_mon'+f'{j}.txt', delimiter=' ',names=["u","i","r"])
    # 348726
    for i in range(348726):
        r = df['r'][i]
        rating.append(r)




rating_set = [rating[i:i + 348726] for i in range(0, len(rating), 348726)]
rating_array=np.array(rating_set)
print(type(rating_array))
print(np.shape(rating_array))


np.save('item_adcos_prediction_test_movielen1m.npy',rating_array)