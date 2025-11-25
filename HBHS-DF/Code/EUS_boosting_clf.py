from os import replace
import numpy as np
import sklearn
from collections import Counter
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import csv
from imblearn.metrics import geometric_mean_score

warnings.filterwarnings("ignore")


class EUS_boosting(BaseEstimator, ClassifierMixin):
    base_estimator = DecisionTreeClassifier()

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_estimators=10,
                 k=1,
                 random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.k = k
        self.random_state = random_state
        self.estimators_ = []

        self.weight_ = []

    @classmethod
    def fit_base_estimator(self, X, y):

        """Private function used to train a single base estimator."""
        return sklearn.base.clone(self.base_estimator).fit(X, y)

    def fit(self,x,y,generated_x,label_maj=0,label_min=1,):
        x_maj = x[y == label_maj]
        y_maj = y[y == label_maj]
        x_min = x[y == label_min]
        y_min = y[y == label_min]
        maj_idx = [idx for idx in range(x.shape[0]) if y[idx] == label_maj]

        self.bin_num = x_min.shape[0]
        self.y_pred_maj = np.zeros(x_maj.shape[0])

        for i_estimator in range(0, self.n_estimators):
            clf=self.fit_estimator(x_maj,x_min,y_maj,y_min,generated_x)
            inter_prob=clf.predict_proba(x)
            inter_pred = inter_prob.argmax(axis=1)
            g_mean_ma = geometric_mean_score(y, inter_pred, average='macro')
            self.weight_.append(g_mean_ma)

            W = np.array(self.weight_)
            temp_maj_prob = inter_prob[maj_idx, 0]
            self.y_pred_maj = self.y_pred_maj * (W[0:-1].sum() / W.sum()) + temp_maj_prob * (W[-1] / W.sum())

            self.estimators_.append(clf)

        return

    def predict(self, x):
        y_prob = self.predict_proba(x)

        y_pred = y_prob.argmax(axis=1)

        return y_pred

    def predict_proba(self, x):
        w = np.array(self.weight_)
        w = w / w.sum()
        #        print(f"w_len:{w},#estimators:{len(self.estimators_)}")
        y_prob = np.array(
            [model.predict_proba(x) * w[i] for i, model in enumerate(self.estimators_)]
        ).sum(axis=0)

        return y_prob

    def fit_estimator(self,x_maj,x_min,y_maj,y_min,generated_x):
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

    def equalization_sampling(self):
        prob = self.y_pred_maj
        bin_num = self.bin_num

        step = (prob.max() - prob.min()) / bin_num
        interval = np.arange(prob.min(), prob.max(), step)
        part_bin = np.digitize(prob,
                                    interval)
        part_cnt = Counter(
            part_bin)

        noempty_bin_key = [i for i in part_cnt.keys()]
        noempty_bin_key.sort()

        s = np.zeros(bin_num + 2)
        s[noempty_bin_key] = 1  
        s = s * (1.0 / len(part_cnt)) * bin_num * self.k
        s = np.ceil(s).astype(int)

        sampled_bins = []

        start_s_index = 1
        pre_key = noempty_bin_key[0]
        cur_needs = 0

        for key in noempty_bin_key:
            temp_elements = []
            ele_cnt = part_cnt[key]
            cur_needs = s[start_s_index:key].sum()
            start_s_index = key + 1  # record the next bin which will be sampled.
            # The bins from pre_index to key-1 need to sample, but there aren't enough samples in them.
            if cur_needs > 0:
                if pre_key == 0:  # pre_key is the last bin,need  to start from the first element in part_bin
                    temp_elements = np.where(part_bin == (len(noempty_bin_key) - 1))[-1].tolist()
                else:
                    temp_elements = np.where(part_bin == pre_key)[-1].tolist()
                sampled_bins.append(np.random.choice(
                    temp_elements, cur_needs, replace=True))

            if s[key] <= ele_cnt:  # the number of the samples in currrent bin are greater than that of needing samples.
                sampled_bins.append(np.random.choice(
                    np.where(part_bin == key)[-1].tolist(),
                    s[key],
                    replace=False))
            else:
                temp_elements += np.where(part_bin == key)[-1].tolist()
                sampled_bins.append(np.random.choice(
                    temp_elements, s[key], replace=True))
            pre_key = key
        # remaining bins
        if start_s_index < bin_num:
            cur_needs = s[start_s_index:bin_num].sum()
            temp_elements = np.where(part_bin == pre_key)[-1].tolist()
            sampled_bins.append(np.random.choice(
                temp_elements, cur_needs, replace=True))

        return sampled_bins
