import numpy as np
from gcForest import gcForest
import time
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, \
    roc_auc_score, average_precision_score
from imblearn.metrics import geometric_mean_score
import csv
from evaluation import f1_macro

def get_predict_report(y_test, y_pred, y_prob):
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = report['accuracy']
    recall_0 = report["0"]['recall']
    recall_1 = report["1"]['recall']
    precision_0 = report["0"]['precision']
    precision_1 = report["1"]['precision']

    f1 = f1_score(y_test, y_pred)
    f1_ma = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred)
    recall_ma = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred)
    precision_ma = precision_score(y_test, y_pred, average='macro')
    g_mean = geometric_mean_score(y_test, y_pred, average='binary')
    g_mean_ma = geometric_mean_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_prob[:, 1])
    aupr = average_precision_score(y_test, y_prob[:, 1])

    pre_report = np.array(
        [acc, recall_0, recall_1, precision_0, precision_1, f1, f1_ma, recall, recall_ma, precision, precision_ma,
         g_mean, g_mean_ma, auc, aupr])

    return pre_report

def output_report(dataset_name,imb_ratio,k,fold,report,ss):
    header = ["acc", " recall_0", "recall_1", "precision_0", "precision_1", "f1", "f1_macro", "recall",
              "recall_macro", "precision", "precision_macro", "g_mean", "g_mean_macro", "auc", "aupr", "time"]
    if fold == 0:
        with open(f'{result_path}/{dataset_name}_IR={imb_ratio}_k={k}_{ss}.csv', mode='w', newline='', encoding='utf8') as cf:
            wf = csv.writer(cf)
            wf.writerow(header)
            wf.writerow(report)
    else:
        with open(f'{result_path}/{dataset_name}_IR={imb_ratio}_k={k}_{ss}.csv', mode='a', newline='', encoding='utf8') as cf:
            wf = csv.writer(cf)
            wf.writerow(report)

    cf.close()

    return

def get_config(k):
    config = {}
    config["random_state"] = 0
    config["max_layers"] = 2
    config["early_stop_rounds"] = 2
    config["if_stacking"] = False
    config["if_save_model"] = False
    config["train_evaluation"] = f1_macro  ##f1_binary,f1_macro,f1_micro
    config["estimator_configs"] = []

    config["enhanced_feature_type"] = "y_prob"
    config["generated_sample_type"]="hardness_depth"
    config["generated_samples_k"]=k
    config["depth_interval"]=0.05
    config["depth_acc_threshold"]=0.7

    for i in range(2):
        config["estimator_configs"].append(
            {"n_fold": 5, "type": "EUS_C", "k": k, "n_jobs": 10})
        config["estimator_configs"].append(
            {"n_fold": 5, "type": "EUS_R", "k": k, "n_jobs": 10})

    return config

K_dict={'Human':{5:2,
           10:2,
           15:3,
           20:3,
           25:3},
        'BioSNAP':{5:3,
           10:3,
           15:5,
           20:5,
           25:5},
        'Drugbank':{5:2,
           10:3,
           15:4,
           20:5,
           25:5}}
IR_l=[5,10,15,20,25]
n_fold = 5
data_path='../../Data/tabular DTI dataset'
n_fold_path= './n_fold/'
result_path='./output'
dataset_list=['Human','BioSNAP','Drugbank']

if __name__=='__main__':
    for imb_ratio in IR_l:
            for dataset_name in dataset_list:
                k = K_dict[dataset_name][imb_ratio]

                config = get_config(k)

                feature = np.load(
                    f'{data_path}/{dataset_name}/{dataset_name}_IR={imb_ratio}_feature.npy')
                label = np.load(
                    f'{data_path}/{dataset_name}/{dataset_name}_IR={imb_ratio}_label.npy')
                label = label.astype(int)

                feature1 = feature[:, 0:11]
                feature2 = feature[:, 15:18]
                feature3 = feature[:, 26:]
                feature = np.hstack((feature1, feature2, feature3))

                feature = np.hstack((feature[:, 0:30], feature[:, 31:]))

                for fold in range(n_fold):
                    train_id = np.load(f'{n_fold_path}/{dataset_name}/IR={imb_ratio}_{dataset_name}_train_fold{fold}_id.npy')
                    test_id = np.load(f'{n_fold_path}/{dataset_name}/IR={imb_ratio}_{dataset_name}_test_fold{fold}_id.npy')

                    x_train, x_test = feature[train_id], feature[test_id]
                    y_train, y_test = label[train_id], label[test_id]

                    print(
                        f'===fold_{fold} IR={imb_ratio} dataset={dataset_name}  ===========')

                    gcf = gcForest(config)
                    t1 = time.time()
                    gcf.fit(x_train, y_train)
                    y_pred = gcf.predict(x_test)
                    y_prob=gcf.predict_proba(x_test)
                    t2 = time.time()

                    test_report = get_predict_report(y_test, y_pred, y_prob)
                    test_report = np.append(test_report, t2 - t1)

                    output_report(dataset_name, imb_ratio, k, fold,
                                  test_report, "test")
