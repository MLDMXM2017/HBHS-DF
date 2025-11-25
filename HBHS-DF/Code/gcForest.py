import numpy as np
from layer import layer
from logger import get_logger
from k_fold_wrapper import KFoldWapper
from EUS_R_clf import EUS_R
from EUS_C_clf import EUS_C
from sklearn.metrics import accuracy_score
import csv
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, \
    roc_auc_score, average_precision_score
from imblearn.metrics import geometric_mean_score

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

LOGGER=get_logger("gcForest")

class gcForest(object):
    def __init__(self,config):
        self.random_state = config["random_state"]
        self.max_layers = config["max_layers"]
        self.early_stop_rounds = config["early_stop_rounds"]
        self.if_stacking = config["if_stacking"]
        self.if_save_model = config["if_save_model"]
        self.train_evaluation = config["train_evaluation"]
        self.estimator_configs = config["estimator_configs"]
        self.layers = []

        self.enhanced_feature_type = config["enhanced_feature_type"]
        self.generated_sample_type=config["generated_sample_type"]
        self.generated_samples_k=config["generated_samples_k"]
        self.depth_interval=config["depth_interval"]
        self.depth_acc_threshold=config["depth_acc_threshold"]

    def fit(self, x_train, y_train):
        x_train, n_feature, n_label = self.preprocess(x_train, y_train)

        evaluate = self.train_evaluation
        best_layer_id = 0
        best_layer_evaluation = 0.0
        best_layer_label_temp = None
        best_layer_prob = None

        enhanced_feature=None
        sample_candidate_idx=[]
        depth=0

        while depth<self.max_layers:
            current_layer = layer(depth)
            LOGGER.info(
                "-----------------------------------------layer-{}--------------------------------------------".format(
                    current_layer.layer_id))
            LOGGER.info("The shape of x_train is {}".format(x_train.shape))

            y_train_probas = np.zeros((x_train.shape[0], n_label * len(self.estimator_configs)))
            y_train_probas_avg = np.zeros((x_train.shape[0], n_label))

            for index in range(len(self.estimator_configs)):
                config = self.estimator_configs[index].copy()
                k_fold_est = KFoldWapper(current_layer.layer_id, index, config, random_state=self.random_state)

                y_proba = k_fold_est.fit(x_train, y_train, sample_candidate_idx, self.generated_samples_k)
                current_layer.add_est(k_fold_est)
                y_train_probas[:, index * n_label:index * n_label + n_label] += y_proba
                y_train_probas_avg += y_proba

            y_train_probas_avg /= len(self.estimator_configs)
            label_tmp = self.category[np.argmax(y_train_probas_avg, axis=1)]
            current_evaluation = evaluate(y_train, label_tmp)


            if current_evaluation > best_layer_evaluation + 0.001:
                best_layer_id = current_layer.layer_id
                best_layer_evaluation = current_evaluation
                best_layer_label_temp = label_tmp
                best_layer_prob = y_train_probas_avg
            LOGGER.info(
                "The evaluation[{}] of layer_{} is {:.4f}".format(evaluate.__name__, depth, current_evaluation))

      
            self.layers.append(current_layer)


            if current_layer.layer_id - best_layer_id >= self.early_stop_rounds:
                self.layers = self.layers[0:best_layer_id + 1]
                LOGGER.info("training finish...")
                LOGGER.info(
                    "best_layer: {}, current_layer:{}, save layers: {}".format(best_layer_id, current_layer.layer_id,
                                                                               len(self.layers)))
                break

            depth += 1

            sample_hardness = None
  
            if self.generated_sample_type == "hardness_depth":
                sample_hardness = self.get_train_sample_hardness(current_layer, x_train)

        
            if self.enhanced_feature_type=='y_prob':
                enhanced_feature = np.copy(y_train_probas)
            elif self.enhanced_feature_type=='hardness_depth':
                if sample_hardness is None:
                    sample_hardness = self.get_train_sample_hardness(current_layer, x_train)
                enhanced_feature=np.copy(sample_hardness)

        
            if self.if_stacking:
                x_train = np.hstack((x_train, enhanced_feature))
            else:
                x_train = np.hstack((x_train[:, 0:n_feature], enhanced_feature))

            if self.generated_sample_type=="hardness_depth":
                sample_hardness=np.average(sample_hardness,axis=1)
       
                sample_candidate_idx =self.find_candidate_sample_idx(y_train,label_tmp,sample_hardness)

        return

    def predict(self, x):
        prob = self.predict_proba(x)
        label = self.category[np.argmax(prob, axis=1)]
        return label


    def predict_proba(self,x_test):
        n_feature = x_test.shape[1]

        n_label = 2
        enhanced_feature=None
        for depth in range(len(self.layers)):
            y_test_probas = np.zeros((x_test.shape[0], n_label * len(self.estimator_configs)))
            y_test_probas_avg = np.zeros((x_test.shape[0], n_label))

            for est_idx in range(len(self.estimator_configs)):
                y_proba = self.layers[depth].estimators[est_idx].predict_proba(x_test)

                y_test_probas[:, est_idx * n_label:est_idx * n_label + n_label] += y_proba
                y_test_probas_avg += y_proba

            y_test_probas_avg /= len(self.estimator_configs)


            sample_hardness = self.get_test_sample_hardness(depth, x_test)

            if self.enhanced_feature_type == 'y_prob':
                enhanced_feature=np.copy(y_test_probas)
            elif self.enhanced_feature_type=='hardness_depth':
                enhanced_feature=np.copy(sample_hardness)

            if (not self.if_stacking):
                x_test = x_test[:, 0:n_feature]
            x_test = np.hstack((x_test, enhanced_feature))

        return y_test_probas_avg

    def preprocess(self,x_train,y_train):
        x_train=x_train.reshape((x_train.shape[0],-1))
        category=np.unique(y_train)
        self.category=category
        #print(len(self.category))
        n_feature=x_train.shape[1]
        n_label=len(np.unique(y_train))
        LOGGER.info("Begin to train....")
        LOGGER.info("the shape of training samples: {}".format(x_train.shape))
        LOGGER.info("use {} as training evaluation".format(self.train_evaluation.__name__))
        LOGGER.info("stacking: {}, save model: {}".format(self.if_stacking,self.if_save_model))
        return x_train,n_feature,n_label

    def get_train_sample_hardness(self,layer,x):
        sample_hardness=None

        for idx in range(len(self.estimator_configs)):
            k_fold_est=layer.estimators[idx]
            fold_hardness=None
            for fold in range(k_fold_est.n_fold):
                valid_id=k_fold_est.cv[fold][1]
                hardness=self.get_clf_hardness(k_fold_est.estimators[fold],x[valid_id])
                if fold==0:
                    fold_hardness=np.zeros((x.shape[0],hardness.shape[1]))
                fold_hardness[valid_id]+=hardness

            if idx==0:
                sample_hardness=np.copy(fold_hardness)
            else:
                sample_hardness=np.hstack([sample_hardness,fold_hardness])

        return sample_hardness

    def get_test_sample_hardness(self,layer_idx,x):
        sample_hardness = None

        for idx in range(len(self.estimator_configs)):
            k_fold_est = self.layers[layer_idx].estimators[idx]
            fold_hardness=None
            for fold in range(k_fold_est.n_fold):
                hardness = self.get_clf_hardness(k_fold_est.estimators[fold], x)
                if fold == 0:
                    fold_hardness=np.copy(hardness)
                else:
                    fold_hardness+=hardness

            fold_hardness/=k_fold_est.n_fold
            if idx==0:
                sample_hardness=np.copy(fold_hardness)
            else:
                sample_hardness=np.hstack([sample_hardness,fold_hardness])

        return sample_hardness

    def get_clf_hardness(self,clf,x):
        hardness_arr=None
        if isinstance(clf, EUS_R):
            hardness_arr=np.zeros((x.shape[0],clf.bagging_n_estimators))
            for est_idx in range(clf.bagging_n_estimators):
                dt=clf.estimators_[-1][est_idx]
                path = np.array(dt.decision_path(x).A)
                path_len = np.sum(path, axis=1)

                max_depth = dt.tree_.max_depth
                hardness = path_len / max_depth

                hardness_arr[:, est_idx] = hardness

        elif isinstance(clf, EUS_C):
            hardness_arr=np.zeros((x.shape[0],clf.bagging_n_estimators))
            for est_idx in range(clf.bagging_n_estimators):
                k_ease=clf.estimators_[est_idx]
                dt=k_ease.estimators_[-1]
                path=np.array(dt.decision_path(x).A)
                path_len=np.sum(path,axis=1)

                max_depth=dt.tree_.max_depth
                hardness=path_len/max_depth

                hardness_arr[:,est_idx]=hardness

        return hardness_arr

    def find_candidate_sample_idx(self,label,y_pred,sample_hardness):
        interval_acc_dict={}

        for d in np.arange(0,1,self.depth_interval):
            sample_idx=[idx for idx,depth in enumerate(sample_hardness) if depth>=d and depth < d + self.depth_interval and label[idx]==1]
            sub_y=label[sample_idx]
            sub_pred=y_pred[sample_idx]

            acc = accuracy_score(sub_y, sub_pred)
            interval_acc_dict[d]=acc

        candidate_idx=[]
        for d in interval_acc_dict.keys():
            if interval_acc_dict[d]<self.depth_acc_threshold:
                print(f"The picked depth threshold={d}")
                candidate_idx=[idx for idx,depth in enumerate(sample_hardness) if depth>=d and label[idx]==1]
                break

        return candidate_idx










