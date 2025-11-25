from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
from EUS_boosting_clf import EUS_boosting
import numpy as np

class EUS_C:
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 bagging_n_estimators=10,
                 boosting_num=10,
                 random_state=None,
                 n_jobs=-1,
                 k=1):
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.bagging_n_estimators = bagging_n_estimators
        self.boosting_num = boosting_num
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.k = k
        self.weight_ = []

    def fit(self,x,y,generated_x,label_maj=0,label_min=1):
        self.estimators_.extend(Parallel(n_jobs=self.n_jobs)(
            delayed(self.multi_process_fit)(x,y,generated_x) for _ in range(0, self.bagging_n_estimators)))

        return

    def predict(self,x):
        y_prob = self.predict_proba(x)

        y_pred = y_prob.argmax(axis=1)

        return y_pred

    def predict_proba(self,x):
        y_prob = np.zeros((x.shape[0], 2))

        for bagging_idx in range(self.bagging_n_estimators):
            y_prob+=self.estimators_[bagging_idx].predict_proba(x)
        y_prob/= self.bagging_n_estimators

        return y_prob

    def multi_process_fit(self,x,y,generated_x):
        clf=EUS_boosting(base_estimator=self.base_estimator, n_estimators=self.boosting_num, k=self.k, random_state=self.random_state)
        clf.fit(x,y,generated_x)

        return clf



