from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pickle
import numpy as np
import copy

def softmax(X):
    """Compute softmax values for each sets of scores in x: shape = (N, d)."""
    e_X = np.exp(X - np.max(X))
    return e_X / e_X.sum(axis=1).reshape(-1, 1)

def sigmoidal_normalize(prob):
    """Probability estimation for OvR logistic regression.
    Positive class probabilities are computed as
    1. / (1. + np.exp(-self.decision_function(X)));
    multiclass is handled by normalizing that over all classes.
    ** Used by OVR logistic regression implicitly
    """
    prob *= -1
    np.exp(prob, prob)
    prob += 1
    np.reciprocal(prob, prob)
    if prob.ndim == 1:
        return np.vstack([1 - prob, prob]).T
    else:
        prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
        return prob

def get_classifiers(X, y, classes, classifier="logistic", kernel="linear", weights=None, random_state=0, ncpu=1):
    """
    Args:
        X : d-dimensional features of shape (N, d)
        y : labels of shape (N,)
        classes : List of classes
    """
    classifiers = []
    for c in classes:
        idxs = y == c
        y_binary = copy.deepcopy(y)
        y_binary[idxs] = 1
        y_binary[~idxs] = 0
        if classifier == "logistic":
            clf = LogisticRegression(random_state=random_state+c, n_jobs=ncpu)
        elif classifier == "svc":
            clf = SVC(kernel=kernel, probability=True, random_state=random_state+c)
        elif classifier == "linearsvc":
            # NOTE: dual is set to False
            clf = CalibratedClassifierCV(LinearSVC(dual=False, random_state=random_state+c))
        clf = clf.fit(X, y_binary, sample_weight=weights)
        classifiers.append(clf)
    return classifiers

def get_ovr_classifier(X, y, classifier="logistic", kernel="linear", weights=None, random_state=0, ncpu=1):
    """
    One-Vs-Rest
    """
    if classifier == "logistic":
        clf = LogisticRegression(multi_class='ovr', random_state=random_state, n_jobs=ncpu)
    elif classifier == "svc":
        clf = SVC(kernel=kernel, probability=True, decision_function_shape='ovr', random_state=random_state+c)
    elif classifier == "linearsvc":
        # NOTE: dual is set to False
        clf = CalibratedClassifierCV(LinearSVC(dual=False, multi_class='ovr', random_state=random_state+c))
    clf = clf.fit(X, y, sample_weight=weights)
    return clf

if __name__ == '__main__':
    x = np.random.rand(100, 10)
    y = np.random.randint(10, size=(100))
