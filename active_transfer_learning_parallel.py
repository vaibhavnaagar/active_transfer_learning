from __future__ import print_function, division
import numpy as np
import pickle
from itertools import combinations
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
import cvxopt
import argparse
from plotter import Plotter
from calc_sigma import compute_sigma
import multiprocessing as mp

# import custom classifiers #
from construct_classifiers import get_classifiers, softmax, sigmoidal_normalize

parser = argparse.ArgumentParser(description='Active Transfer Learning with Cross-class Similarity Transfer')
parser.add_argument('--dset', '-d', required=True, help='Path to dataset')
parser.add_argument('--G', '-g', required=True, help='Path to class similarity matrix')
parser.add_argument('--model', '-m', default='alexnet', help='Model used to construct feature vectors')
parser.add_argument('--nlabels', '-l', type=int, default=10, help='Number of labels or classes in dataset')
parser.add_argument('--workers', '-w', type=int, default=1, help='Number of CPU cores')
parser.add_argument('--sigma', '-s', type=float, default=0., help='Sigma for heat kernel similarity')
args = parser.parse_args()

# CIFAR10 #
# classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classes = list(range(args.nlabels))
num_source_classes = 8
num_target_classes = 2
unlabeled_data_size = 0.999666
ncpu = args.workers
max_iterations = 1
n_expert_samples = 2         # Number of samples selected from unlabeled data for expert labeling
n_random_samples_TL = 500       # Number of random samples needed in Sample-sample similarity graph
n_transfer_samples = 200     # Number of samples transferred from source to target
n_random_samples_AL = 1000     # Number of random samples needed in computing heat kernel similarity for unlabeled samples
lambdaa = 0.5
# tau=0.01 and eta=0.0001
tau = 0.01
eta = 1e-4
if args.sigma is 0.0:
    print("No sigma provided! :(\nComputing Sigma...")
    sigma = compute_sigma(args.dset)
else:
    sigma = args.sigma # 60.608
print("Sigma:", sigma)

def LabelBasedDataSplit(X, y, labels):
    """
    Args:
        X : d-dimensional features of shape (N, d)
        y : labels of shape (N,)
        label : list or tuple of labels
    """
    idxs = np.sum([y==l for l in labels], axis=0).astype(bool)
    return [X[idxs], y[idxs]], [X[~idxs], y[~idxs]]             # Fancy Indexing

def DataSplit(D, ratio=0.5, random_state=0):
    """
    Args:
        D : (X, y) where X = d-dimensional features of shape (N, d) and y = labels of shape (N,)
        ratio : partition ratio (unlabeled data size)
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=random_state)
    split1, split2 = list(sss.split(*D))[0]
    return [D[0][split1], D[1][split1]], [D[0][split2], D[1][split2]]       # Fancy Indexing

def computeProbability(X, classifier, normalize_method="sigmoid"):
    if type(classifier) not in [list, tuple]:
        return classifier.predict_proba(X)
    return normalize(np.hstack([clf.predict_proba(X)[:, 1].reshape(-1, 1) for clf in classifier]),
                     method=normalize_method, return_split=False)

def heatKernelSimilarity(feature_vecs, sigma=None):
    """
    feature_vecs : 2 element list or tuple of feature vectors of shape (N, d)
    return: 1-1, 1-2, 2-1, 2-2
    """
    assert len(feature_vecs) == 2
    idx1 = list(range(feature_vecs[0].shape[0]))
    idx2 = list(range(idx1[-1] + 1, idx1[-1] + 1 + feature_vecs[1].shape[0]))
    fv = np.vstack(feature_vecs)
    # try:
    #     Memory inefficient method
    #     fvs = fv.reshape(fv.shape[0], 1, fv.shape[1])
    #     sq_euclidean_dist = np.einsum('ijk, ijk->ij', fv-fvs, fv-fvs)
    # except MemoryError:
    sq_euclidean_dist = euclidean_distances(fv, fv, squared=True)
    sigma = np.average(np.sqrt(sq_euclidean_dist)) if sigma is None else sigma
    hks = np.exp(-sq_euclidean_dist/(sigma**2))
    return hks[np.ix_(idx1, idx1)], hks[np.ix_(idx1, idx2)], hks[np.ix_(idx2, idx1)], hks[np.ix_(idx2, idx2)], sigma

def heatKernelSimilarity_v2(V1, V2, sigma=None):
    """
    """
    sq_euclidean_dist = euclidean_distances(V1, V2, squared=True)
    sigma = sigma if sigma else np.average(np.sqrt(sq_euclidean_dist))
    hks = np.exp(-sq_euclidean_dist/(sigma**2))
    return hks

def normalize(*matrices, method="l1", return_split=False):
    """ Row-wise normalization, assumes number of rows in all matrices are same
    """
    if len(matrices) == 0:
        return None
    sizes = [m.shape[1] for m in matrices]
    mat = np.hstack(matrices)
    if method == "l1":
        mat /= mat.sum(axis=1).reshape(-1, 1)
    elif method == "l2":
        mat = mat**2
        mat /= mat.sum(axis=1).reshape(-1, 1)
    elif method == "softmax":
        mat = softmax(mat)
    elif method == "sigmoid":
        mat = sigmoidal_normalize(mat)
    else:
        raise NotImplementedError
    if return_split:
        return np.split(mat, np.cumsum(sizes), axis=1)[:-1]
    return mat

def eval_classifier(classifier, features, true_label, classes=None):
    if type(classifier) not in [list, tuple]:
        predicted_label = classifier.predict(features)
    else:
        probs = computeProbability(features, classifier)
        classes = range(probs.shape[1]) if classes is None else classes
        predicted_label = np.array(classes)[np.argmax(probs, axis=1)]
    acc = accuracy_score(true_label, predicted_label)
    print("Accuracy:", acc*100, "%")
    return acc

class ATL():
    """Active Transfer Learning with Cross-class Similarity Transfer"""
    def __init__(self, G, source_classes, target_classes, sigma, **params):
        self.source_classes = list(source_classes)
        self.target_classes = list(target_classes)
        self.num_target_classes = len(target_classes)
        self.G = G
        self.sigma = sigma
        self.max_iterations = params.get("max_iterations", 20)
        self.u_data_size = params.get("unlabeled_data_size", None)
        self.random_state = params.get("random_state", 0)
        self.ncpu = params.get("ncpu", 1)
        self.lambdaa = params.get("lambdaa", 0.5)
        self.tau = params.get("tau", 0.01)
        self.eta = params.get("eta", 1e-4)
        self.n_random_samples_TL = params.get("n_random_samples_TL", 500)
        self.n_transfer_samples = params.get("n_transfer_samples", 200)
        self.n_random_samples_AL = params.get("n_random_samples_AL", 1000)
        self.n_expert_samples = params.get("n_expert_samples", 2)
        self.overall_acc = 0.0
        self.accuracy_scores = []

    def __call__(self, train_data, train_labels, test_data, test_labels, run_algo=False):
        D_source_test = self.preprocess_data(train_data, train_labels, test_data, test_labels)
        print("Building source classifiers")
        # source_classifier = get_ovr_classifier(*D_s, random_state=i, ncpu=ncpu)
        self.source_classifiers = get_classifiers(*self.D_s, self.source_classes, classifier="logistic",
                                                  random_state=self.random_state, ncpu=self.ncpu)
        print("Validating Source classifier")
        eval_classifier(self.source_classifiers, D_source_test[0], D_source_test[1], classes=self.source_classes)
        del D_source_test

        print("Building target classifiers on all samples")
        dummy_classifiers = get_classifiers(*self.D_p, self.target_classes, classifier="linearsvc",
                                            random_state=self.random_state, ncpu=self.ncpu)
        self.overall_acc = eval_classifier(dummy_classifiers, self.D_t[0], self.D_t[1], classes=self.target_classes)
        del dummy_classifiers
        print("Generating Heat....")
        self.heat_kernel_similarity_matrices()
        if run_algo:
            self.run_algorithm(normalize_method="l1")
        return (self.overall_acc, self.accuracy_scores)

    def class_class_similarity_graph(self, normalize_method="sigmoid"):
        """Class-class similarity graph"""
        print("Class-class similarity graph")
        G_ss = self.G[np.ix_(self.source_classes, self.source_classes)]
        G_st = self.G[np.ix_(self.source_classes, self.target_classes)]
        GG = np.linalg.inv(np.identity(len(self.source_classes)) - G_ss) @ G_st
        src_sim_src = computeProbability(self.D_s[0], self.source_classifiers, normalize_method=normalize_method)
        src_sim_tgt_c = src_sim_src @ GG
        print(self.src_sim_tgt_c.shape)
        return src_sim_tgt_c

    def sample_sample_similarity_graph(self, normalize_method="l1"):
        """Sample-sample similarity graph"""
        print("Sample-sample similarity graph")
        target_indexes = np.arange(self.L_p[0].shape[0])
        source_indexes = np.arange(self.D_s[0].shape[0])
        src_random_samples_idxs = np.random.choice(self.D_s[0].shape[0], self.n_random_samples_TL, replace=False)
        H_ss_ = self.H_ss[np.ix_(src_random_samples_idxs, src_random_samples_idxs)]
        H_st_ = self.H_st[np.ix_(src_random_samples_idxs, target_indexes)]
        H_ss_, H_st_ = normalize(H_ss_, H_st_, method=normalize_method, return_split=True)
        print(H_ss_.shape, H_st_.shape)

        H_ts_ = self.H_ts[np.ix_(target_indexes, src_random_samples_idxs)]
        H_tt_ = self.H_tt.copy()
        Y_tc = np.zeros((self.L_p[0].shape[0], self.num_target_classes))      # One hot encoding
        for col, c in enumerate(self.target_classes):
            Y_tc[self.L_p[1]==c, col] = 1
        H_ts_, H_tt_, Y_tc = normalize(H_ts_, H_tt_, Y_tc, method=normalize_method, return_split=True)
        print(H_ts_.shape, H_tt_.shape, Y_tc.shape)
        H_st_st = np.vstack((np.hstack((H_ss_, H_st_)), np.hstack((H_ts_, H_tt_))))
        print(H_st_st.shape)
        H_st_c = np.vstack((np.zeros((self.n_random_samples_TL, self.num_target_classes)), Y_tc))
        print(H_st_c.shape)
        HH = np.linalg.inv(np.identity(self.n_random_samples_TL + self.L_p[0].shape[0]) - H_st_st) @ H_st_c
        print(HH.shape)

        H_xs = self.H_ss[np.ix_(source_indexes, src_random_samples_idxs)]
        H_xt = self.H_st.copy()
        H_xs, H_xt = normalize(H_xs, H_xt, method=normalize_method, return_split=True)
        print(H_xs.shape, H_xt.shape)
        src_sim_tgt_s = np.hstack((H_xs, H_xt)) @ HH
        print(src_sim_tgt_s.shape)
        return src_sim_tgt_s

    def run_algorithm(self, normalize_method="l1"):
        p_ic = self.class_class_similarity_graph()
        transferred_samples = None
        replace = True      # Starts with replace true to discard randomly chosen 2 samples in labeled set
        print("Let's begin!")
        for i in range(self.max_iterations):
            print("#%d" % i)
            # Update Heat Kernel similarity matrix #
            if transferred_samples is not None:
                if replace:
                    replace = False
                    self.H_st = heatKernelSimilarity_v2(self.D_s[0], transferred_samples, sigma=self.sigma)
                    self.H_ts = self.H_st.T        # XXX: Not really required
                    self.H_tt = heatKernelSimilarity_v2(self.L_p[0], transferred_samples, sigma=self.sigma)
                else:
                    self.H_st = np.hstack((self.H_st, heatKernelSimilarity_v2(self.D_s[0], transferred_samples, sigma=self.sigma)))
                    self.H_ts = self.H_st.T        # XXX: Not really required
                    H_tts = heatKernelSimilarity_v2(self.L_p[0], transferred_samples, sigma=self.sigma)
                    self.H_tt = np.hstack((self.H_tt, H_tts[:-transferred_samples.shape[0]]))
                    self.H_tt = np.vstack((self.H_tt, H_tts.T))
                print("HeatKernelSimilarity Updated:", self.H_ss.shape, self.H_st.shape, self.H_ts.shape, self.H_tt.shape)

            # Combine similarities between source samples to target classes from both graphs #
            p_is = self.sample_sample_similarity_graph(normalize_method=normalize_method)
            src_sim_tgt = self.lambdaa * p_ic + (1 - self.lambdaa) * p_is
            print(src_sim_tgt.shape)

            self.construct_target_classifier(src_sim_tgt)
            self.accuracy_scores.append(eval_classifier(self.target_classifiers, *self.D_t, classes=self.target_classes))

            unlabeled_ranking_scores = self.compute_rankings(normalize_method="softmax")
            transferred_samples= self.augment_labeled_set(unlabeled_ranking_scores, replace=replace)
            print("Iteration #%d completed!" % it)
        return

    def augment_labeled_set(self, R_p, replace=False):
        """ Augment Labeled set by Expert Labeling """
        print("Expert Labeling")
        u_idx = np.argpartition(R_p, -self.n_expert_samples)[-self.n_expert_samples:]
        print("Now let's see the ranking of top %d unlabeled samples:" % self.n_expert_samples, R_p[u_idx], u_idx)
        transferred_samples = self.U_p[0][u_idx].copy()
        if replace:
            self.L_p[0] = transferred_samples.copy()
            self.L_p[1] = self.U_p[1][u_idx].copy()
        else:
            self.L_p[0] = np.vstack((self.L_p[0], transferred_samples))
            self.L_p[1] = np.vstack((self.L_p[1].reshape(-1,1), self.U_p[1][u_idx].reshape(-1,1))).reshape(-1)
        self.U_p[0], self.U_p[1] = np.delete(self.U_p[0], u_idx, axis=0), np.delete(self.U_p[1], u_idx, axis=0)
        print("Updated labeled and unlabeled data:")
        print(self.L_p[0].shape, self.L_p[1].shape, self.U_p[0].shape, self.U_p[1].shape)
        self.K_uu = np.delete(np.delete(self.K_uu, u_idx, axis=0), u_idx, axis=1)
        print("Updated K_uu:", self.K_uu.shape)
        return transferred_samples

    def compute_rankings(self, normalize_method="softmax"):
        """Ranking score of unlabeled samples by solving the convex optimization problem """
        # Entropy computation on unlabeled target data #
        print("Computing Entropy on unlabeled target data")
        U_sim_tgt = computeProbability(self.U_p[0], self.target_classifiers, normalize_method=normalize_method)
        E_u = -np.sum(U_sim_tgt * np.log(U_sim_tgt), axis=1).reshape(-1, 1)
        print(E_u.shape)

        src_rs_idxs = np.random.choice(self.D_s[0].shape[0], self.n_random_samples_AL, replace=False)
        K_us = heatKernelSimilarity_v2(self.U_p[0], self.D_s[0][src_rs_idxs], sigma=self.sigma)
        print("HeatKernelSimilarity of unlabeled data:", self.K_uu.shape, K_us.shape)

        print("Ranking score of unlabeled samples by solving the convex optimization problem")
        # NOTE: multiply by 2 as in paper quadratic term is not multiplied by half
        P = cvxopt.matrix((2 * self.eta * self.K_uu).astype(np.double))
        q = cvxopt.matrix(-((self.K_uu @ E_u) + self.tau*(K_us @ np.ones(shape=(self.n_random_samples_AL, 1)))).astype(np.double))
        G = cvxopt.matrix((0.0 - np.identity(self.K_uu.shape[0])).astype(np.double))
        h = cvxopt.matrix(0.0, (self.K_uu.shape[0], 1))
        A = cvxopt.matrix(1.0, (1, self.K_uu.shape[0]))
        b = cvxopt.matrix(1.0)
        R_p = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x']).reshape(-1)
        print("Ranking matrix:", R_p.shape)
        return R_p

    def preprocess_data(self, train_data, train_labels, test_data, test_labels):
        print("Splitting data based on labels")
        self.D_p, self.D_s = LabelBasedDataSplit(train_data, train_labels, self.target_classes)
        self.D_t, D_source_test = LabelBasedDataSplit(test_data, test_labels, self.target_classes)
        target_data = (np.vstack((self.D_p[0], self.D_t[0])),
                       np.vstack((self.D_p[1].reshape(-1,1), self.D_t[1].reshape(-1,1))).reshape(-1)
                      )
        # Splitting target data equally into train and test #
        self.D_p, self.D_t = DataSplit(target_data, ratio=0.5, random_state=self.random_state)
        print(self.D_p[0].shape, self.D_p[1].shape, self.D_s[0].shape, self.D_s[1].shape, self.D_t[0].shape, self.D_t[1].shape)
        print("Splitting target data into labeled and unlabeled set")
        self.u_data_size = (self.D_p[0].shape[0] - 2)/self.D_p[0].shape[0] if self.u_data_size is None else self.u_data_size
        self.L_p, self.U_p = DataSplit(self.D_p, ratio=self.u_data_size, random_state=self.random_state)
        print(self.L_p[0].shape, self.L_p[1].shape, self.U_p[0].shape, self.U_p[1].shape)
        return D_source_test

    def heat_kernel_similarity_matrices(self):
        self.H_ss, self.H_st, self.H_ts, self.H_tt, _ = heatKernelSimilarity([self.D_s[0], self.L_p[0]], sigma=self.sigma)
        print("HeatKernelSimilarity:", self.H_ss.shape, self.H_st.shape, self.H_ts.shape, self.H_tt.shape)
        self.K_uu = heatKernelSimilarity_v2(self.U_p[0], self.U_p[0], sigma=self.sigma)
        print("Unlabeld HeatKernelSimilarity:", self.K_uu.shape)

    def construct_target_classifier(self, src_similarity, classifier="linearsvc"):
        """ Construct classifiers on target classes """
        print("Expanding Labeled Set by adding top related source samples")
        # Expand Labeled Set by adding top related source samples #
        indexes = []
        weights = []
        transfer_labels = []
        for col, c in enumerate(self.target_classes):
            idx = np.argpartition(src_similarity[:, col], -self.n_transfer_samples)[-self.n_transfer_samples:]
            weights += list(src_sim_tgt[:, col][idx])
            indexes += list(idx)
            transfer_labels += [c] * self.n_transfer_samples

        print("Number of transferred samples: %d" % (len(indexes)))
        print(len(list(set(list(indexes)))))
        expanded_set_L = (np.vstack((self.L_p[0], self.D_s[0][indexes])),
                          np.vstack((self.L_p[1].reshape(-1,1), np.array(transfer_labels).reshape(-1,1))).reshape(-1)
                         )
        L_weights = np.vstack((np.ones(shape=(self.L_p[1].shape[0], 1)), np.array(weights).reshape(-1,1))).reshape(-1)
        print("Expanded Set L:", expanded_set_L[0].shape, expanded_set_L[1].shape, L_weights.shape)

        print("Constructing classifiers on target classes")
        # self.target_classifiers = get_ovr_classifier(*expanded_set_L, classifier=classifier, kernel="linear",
                                            #    weights=L_weights, random_state=self.random_state, ncpu=self.ncpu)
        self.target_classifiers = get_classifiers(*expanded_set_L, self.target_classes, classifier=classifier,
                                             weights=L_weights, random_state=self.random_state, ncpu=self.ncpu)

average_acc = np.empty(shape=(0, max_iterations))
overall_acc = []
def assemble_acc(results):
    global overall_acc, average_acc
    print(results)
    overall_acc.append(results[0])
    average_acc = np.vstack((average_acc, np.array(results[1])))

def generate_plots():
    global average_acc, overall_acc
    print("Generating Plots...")
    average_acc = np.average(average_acc, axis=0).reshape(-1,1)
    overall_acc = np.ones(shape=(max_iterations, 1)) * np.average(overall_acc)
    average_acc_plot = Plotter("plots/cifar10_%s_atl.jpeg" % args.model, num_lines=2, legends=["All samples", "ATL with Cross-class similarity transfer"],
                                xlabel="Number of iterations", ylabel="Accuracy (%)", title="Accuracy vs Iterations" )
    iters = np.arange(max_iterations).reshape(-1,1)
    average_acc_plot(np.hstack((iters, overall_acc)), np.hstack((iters, average_acc)))
    # average_acc_plot.queue.put(None)
    average_acc_plot.queue.join()
    average_acc_plot.clean_up()

if __name__ == '__main__':
    ## Class-Class similarity ##
    with open(args.G, 'rb') as f:
        G = pickle.load(f)
    # Normalize G #
    G = G**2
    G /= G.sum(axis=1).reshape(-1, 1)

    with open(args.dset, 'rb') as f:
        data = pickle.load(f)
        train_data = data['train_features']
        train_labels = data['train_labels'].reshape(-1)
        test_data = data['test_features']
        test_labels = data['test_labels'].reshape(-1)

    atl_pool = mp.Pool()

    for i, target_classes in enumerate(combinations(classes, num_target_classes)):
        print("===========================================")
        print("Combination #%d" % i)
        source_classes = [c for c in classes if c not in target_classes]
        print("Source classes:", source_classes)
        print("Target classes:", target_classes)
        atl = ATL(G, source_classes, target_classes, sigma,
                random_state = i,
                max_iterations = max_iterations,
                unlabeled_data_size = unlabeled_data_size,
                ncpu = ncpu,
                lambdaa = lambdaa,
                tau = tau,
                eta = eta,
                n_random_samples_TL = n_random_samples_TL,
                n_transfer_samples = n_transfer_samples,
                n_random_samples_AL = n_random_samples_AL,
                n_expert_samples = n_expert_samples
               )
        atl_pool.apply_async(atl, args=(train_data, train_labels, test_data, test_labels, True), callback=assemble_acc)
    atl_pool.close()
    atl_pool.join()
    generate_plots()
