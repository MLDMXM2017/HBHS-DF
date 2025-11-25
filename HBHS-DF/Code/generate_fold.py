import numpy as np
from Crypto.Random.random import shuffle
from sklearn.model_selection import StratifiedKFold

dataset_list=["BioSNAP","Drugbank","Human"]
IR=[5,10,15,20,25]
dataset_path='../../Data/tabular DTI dataset'
n_fold_path='./n_fold/'
n_fold=5

if __name__=='__main__':
    for imb_ratio in IR:

        for dataset_name in dataset_list:
            feature_arr = np.load(
                f"{dataset_path}/{dataset_name}/{dataset_name}_IR={imb_ratio}_feature.npy")
            label_arr= np.load(
                f"{dataset_path}/{dataset_name}/{dataset_name}_IR={imb_ratio}_label.npy")

            skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            cv2 = [(t, v) for (t, v) in skf2.split(feature_arr, label_arr)]
            for k in range(n_fold):
                train_id, test_id = cv2[k]
                print(f'train_size={len(train_id)}, test_id={len(test_id)}')

                np.save(n_fold_path + f"{dataset_name}/IR={imb_ratio}_{dataset_name}_train_fold{k}_id.npy", train_id)
                np.save(n_fold_path + f"{dataset_name}/IR={imb_ratio}_{dataset_name}_test_fold{k}_id.npy", test_id)


