from xgboost import XGBClassifier

class BreastCancerClassifier(XGBClassifier):

    def __init__(self):
        super().__init__(
            n_estimators = 400,
            max_depth = 10,
            learning_rate = 0.05,
            colsample_bytree = 1,
            subsample = 1,
            objective = "binary:logistic",
            eval_metric = "logloss",
            random_state = 42,
            n_jobs = -1
        )