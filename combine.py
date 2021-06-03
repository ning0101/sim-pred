import numpy as np
import tensorflow as tf
import glob

adj = "adj_cos_similarity_itembasec_1m_"
cos = "cosine_similarity_userbasec_1m_"
pearson = "pearson_similarity_userbasec_1m_"


sim_load_data_path = 'ml-100k/ml-100k-similarity-result-c/pearson_similarity_userbasec_100k_mon'
sim_save_data_path = 'dataset/ml-100k-similarity/user_pearson_similarity_movielen100k.npy'

predict_load_data_path = "ml-100k/ml-100k-prediction-result-c/user_pearson_c_100k_mon"
predict_save_data_path = 'dataset/ml-100k-prediction/pearson_prediction_userbase_combine_100k.npy'

sim_load_data_path_1M = 'ml-1m/ml-1m-similarity-result-c/'+cos+'mon'
sim_save_data_path_1M = f'dataset/ml-1m-similarity/cosine_similarity_userbase_combine.npy'

predict_load_data_path_1M = "ml-1m/ml-1m-prediction-result-c/item_adcos_c_1m_mon"
predict_save_data_path_1M = 'dataset/ml-1m-prediction/adcos_prediction_itembase_combine_1m.npy'

def concate(data):
    _ = []
    for i in range(data.shape[0]):        
        y = data[i, i:]
        _.append(y)
    data = np.concatenate(_)
    data = np.array(data)
    return data

def similarity_combine(sim_load_data_path, sim_save_data_path):
    data_set = []

    for i in range(35):
        data = np.load(sim_load_data_path+f'{i}.txt.npy', allow_pickle=True)
        print(data.shape)
        data_set.append(concate(data))
    
    data_set = np.array(data_set)
    np.save(sim_save_data_path, data_set)
    print(data_set.shape)   

def prediction_combine(load_data_path, save_data_path):
    data_set = []

    for i in range(35):
        f = open(load_data_path+f"{i}.txt","r")
        lines = f.read().splitlines()
        rating = []
        for r in lines:
            data = r.split()
            rating.append(int(data[2]))
        data_set.append(rating)

    data_set = np.array(data_set)
    np.save(save_data_path, data_set)
    print(data_set.shape)
    
# similarity_combine(sim_load_data_path_1M, sim_save_data_path_1M)
prediction_combine(predict_load_data_path_1M, predict_save_data_path_1M)