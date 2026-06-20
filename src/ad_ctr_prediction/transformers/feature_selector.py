from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)

class CTRFeatureSelector:

    def __init__(self, k=200):
        self.k = k
        self.selector = None

    def fit_transform(self , X , y):
        actual_k = min(self.k, X.shape[1])

        self.selector = SelectKBest(mutual_info_classif, k=actual_k)
        
        return self.selector.fit_transform(X, y)

    def transform(self, X):

        return self.selector.transform(X)