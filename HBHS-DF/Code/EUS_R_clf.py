import sklearn
from joblib import Parallel
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from imblearn.metrics import geometric_mean_score
from collections import Counter
from joblib import Parallel, delayed

class EUS_R(BaseEstimator, ClassifierMixin):
    base_estimator = DecisionTreeClassifier()

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 bagging_n_estimators=10,
                 boosting_num=10,
                 random_state=None,
                 n_jobs=-1,
                 k=3):
        self.base_estimator = base_estimator
        self.estimators_=[]
        self.bagging_n_estimators = bagging_n_estimators
        self.boosting_num = boosting_num
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.k=k
        self.weight_=[]

    @classmethod
    def fit_base_estimator(self, X, y):
        """Private function used to train a single base estimator."""
        return sklearn.base.clone(self.base_estimator).fit(X, y)

    def fit(self,x,y,generated_x,label_maj=0,label_min=1):
        self.x=x

        x_maj = x[y == label_maj]
        y_maj = y[y == label_maj]
        x_min = x[y == label_min]
        y_min = y[y == label_min]
        maj_idx=[idx for idx in range(x.shape[0]) if y[idx]==label_maj]

        self.bin_num = x_min.shape[0]
        self.y_pred_maj = np.zeros(x_maj.shape[0])

        for inter_idx in range(self.boosting_num):
            clf_l=[]

            if inter_idx!=0:
                self.prepare_equalization()
            clf_l.extend(Parallel(n_jobs=self.n_jobs)(
                delayed(self.multi_process_fit)(x_maj,x_min,y_maj,y_min,generated_x) for _ in range(0,self.bagging_n_estimators)))

            self.estimators_.append(clf_l)

            inter_prob=np.zeros((x.shape[0],2))
            for clf in clf_l:
                inter_prob+=clf.predict_proba(x)

            # for idx in range(self.bagging_n_estimators):
            #     new_maj_x,new_maj_y=self.under_sample_maj(x_maj,y_maj)
            #     clf=self.fit_base_estimator(np.vstack([new_maj_x, x_min]), np.hstack([new_maj_y, y_min]))
            #     clf_l.append(self.base_estimator)
            #     inter_prob+=clf.predict_proba(x)

            inter_prob/=self.bagging_n_estimators
            inter_pred=inter_prob.argmax(axis=1)
            g_mean_ma = geometric_mean_score(y, inter_pred, average='macro')
            self.weight_.append(g_mean_ma)
            W = np.array(self.weight_)
            temp_maj_prob=inter_prob[maj_idx,0]
            self.y_pred_maj = self.y_pred_maj * (W[0:-1].sum() / W.sum()) + temp_maj_prob * (W[-1] / W.sum())

        return
    def predict(self,x):
        y_prob=self.predict_proba(x)

        y_pred=y_prob.argmax(axis=1)

        return y_pred

    def predict_proba(self,x):
        y_prob=np.zeros((x.shape[0],2))

        weight_l = np.array(self.weight_)
        weight_l = weight_l / weight_l.sum()

        for inter_idx in range(self.boosting_num):
            inter_prob = np.zeros((x.shape[0], 2))
            for clf in self.estimators_[inter_idx]:
                inter_prob+=clf.predict_proba(x)

            inter_prob /= self.bagging_n_estimators

            y_prob+=inter_prob*weight_l[inter_idx]
        return y_prob

    def multi_process_fit(self,x_maj,x_min,y_maj,y_min,generated_x):
        new_maj_x, new_maj_y = self.under_sample_maj(x_maj, y_maj)

        if generated_x is not None:
            clf = self.fit_base_estimator(np.vstack([new_maj_x, x_min,generated_x]), np.hstack([new_maj_y, y_min,np.ones(generated_x.shape[0],dtype=int)]))
        else:
            clf = self.fit_base_estimator(np.vstack([new_maj_x, x_min]), np.hstack([new_maj_y, y_min]))
        return clf

    def under_sample_maj(self, x_maj, y_maj):
        prob_maj = self.y_pred_maj

        if prob_maj.max() == prob_maj.min():
            maj_idx = self.random_sampling(len(x_maj), self.bin_num)
            new_x_maj = x_maj[maj_idx]
        else:
            maj_sampled_bins = self.equalization_sampling()
            index = np.concatenate(maj_sampled_bins, axis=0)
            new_x_maj = x_maj[index]
        new_y_maj = np.full(new_x_maj.shape[0], y_maj[0])

        return new_x_maj, new_y_maj

    def random_sampling(self, x_len, n):
        np.random.seed(self.random_state)
        idx = np.random.choice(x_len, n, replace=True)
        return idx

    def prepare_equalization(self):
        prob=self.y_pred_maj
        bin_num=self.bin_num

        step = (prob.max() - prob.min()) / bin_num
        interval = np.arange(prob.min(), prob.max(), step)
        self.part_bin = np.digitize(prob,
                               interval)
        self.part_cnt = Counter(
            self.part_bin)

        self.noempty_bin_key = [i for i in self.part_cnt.keys()]
        self.noempty_bin_key.sort()

        s = np.zeros(bin_num + 2)
        s[self.noempty_bin_key] = 1  
        s = s * (1.0 / len(self.part_cnt)) * bin_num * self.k
        self.s = np.ceil(s).astype(int)

        return

    def equalization_sampling(self):
        bin_num=self.bin_num
        sampled_bins = []

        s=np.copy(self.s)

        start_s_index = 1
        pre_key = self.noempty_bin_key[0]
        cur_needs = 0

        for key in self.noempty_bin_key:
            temp_elements = []
            ele_cnt = self.part_cnt[key]
            cur_needs = s[start_s_index:key].sum()
            start_s_index = key + 1  # record the next bin which will be sampled.
            # The bins from pre_index to key-1 need to sample, but there aren't enough samples in them.
            if cur_needs > 0:
                if pre_key == 0:  # pre_key is the last bin,need  to start from the first element in part_bin
                    temp_elements = np.where(self.part_bin == (len(self.noempty_bin_key) - 1))[-1].tolist()
                else:
                    temp_elements = np.where(self.part_bin == pre_key)[-1].tolist()
                sampled_bins.append(np.random.choice(
                    temp_elements, cur_needs, replace=True))

            if s[key] <= ele_cnt:  # the number of the samples in currrent bin are greater than that of needing samples.
                sampled_bins.append(np.random.choice(
                    np.where(self.part_bin == key)[-1].tolist(),
                    s[key],
                    replace=False))
            else:
                temp_elements += np.where(self.part_bin == key)[-1].tolist()
                sampled_bins.append(np.random.choice(
                    temp_elements, s[key], replace=True))
            pre_key = key
        # remaining bins
        if start_s_index < bin_num:
            cur_needs = s[start_s_index:bin_num].sum()
            temp_elements = np.where(self.part_bin == pre_key)[-1].tolist()
            sampled_bins.append(np.random.choice(
                temp_elements, cur_needs, replace=True))

        return sampled_bins