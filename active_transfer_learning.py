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

# import custom classifiers #
from construct_classifiers import get_classifiers, softmax, sigmoidal_normalize

parser = argparse.ArgumentParser(description='Active Transfer Learning with Cross-class Similarity Transfer')
parser.add_argument('--dset', '-d', required=True, help='Path to dataset')
parser.add_argument('--model', '-m', default='alexnet', help='Model used to construct feature vectors')
parser.add_argument('--G', '-g', required=True, help='Path to class similarity matrix')
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
n_random_samples = 500       # Number of random samples needed in Sample-sample similarity graph
n_transfer_samples = 200     # Number of samples transferred from source to target
n_random_samples2 = 1000     # Number of random samples needed in computing heat kernel similarity for unlabeled samples
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

average_acc = np.zeros(shape=(len(list(combinations(classes, num_target_classes))), max_iterations))
overall_acc = []

for i, target_classes in enumerate(combinations(classes, num_target_classes)):
    print("===========================================")
    print("Combination #%d" % i)
    source_classes = [c for c in classes if c not in target_classes]
    print("Source classes:", source_classes)
    print("Target classes:", target_classes)
    G_ss = G[np.ix_(source_classes, source_classes)]
    G_st = G[np.ix_(source_classes, target_classes)]
    GG = np.linalg.inv(np.identity(len(source_classes)) - G_ss) @ G_st
    print("Splitting data based on labels")
    D_p, D_s = LabelBasedDataSplit(train_data, train_labels, target_classes)
    D_t, _ = LabelBasedDataSplit(test_data, test_labels, target_classes)
    target_data = (np.vstack((D_p[0], D_t[0])), np.vstack((D_p[1].reshape(-1,1), D_t[1].reshape(-1,1))))
    # Splitting target data equally into train and test #
    D_p, D_t = DataSplit(target_data, ratio=0.5, random_state=i)
    print(D_p[0].shape, D_p[1].shape, D_s[0].shape, D_s[1].shape, D_t[0].shape, D_t[1].shape)
    print("Splitting target data into labeled and unlabeled set")
    L_p, U_p = DataSplit(D_p, ratio=unlabeled_data_size, random_state=i)
    print("Count Labeled:", np.unique(L_p[1], return_counts=True))
    print("Count Unlabeled:", np.unique(U_p[1], return_counts=True))
    print("Count Test:", np.unique(D_t[1], return_counts=True))
    print(L_p[0].shape, L_p[1].shape, U_p[0].shape, U_p[1].shape)

    print("Building source classifiers")
    source_classifiers = get_classifiers(*D_s, source_classes, classifier="logistic", random_state=i, ncpu=ncpu,)
    eval_classifier(source_classifiers, _[0], _[1], classes=source_classes)
    del _
    # source_classifier = get_ovr_classifier(*D_s, random_state=i, ncpu=ncpu)

    print("Building target classifiers on all samples")
    dummy_classifiers = get_classifiers(*D_p, target_classes, classifier="linearsvc", random_state=i, ncpu=ncpu)
    overall_acc.append(eval_classifier(dummy_classifiers, D_t[0], D_t[1], classes=target_classes))

    H_ss, H_st, H_ts, H_tt, _ = heatKernelSimilarity([D_s[0], L_p[0]], sigma=sigma)
    print("HeatKernelSimilarity:", H_ss.shape, H_st.shape, H_ts.shape, H_tt.shape, "sigma:", sigma)
    K_uu = heatKernelSimilarity_v2(U_p[0], U_p[0], sigma=sigma)
    print("Unlabeld HeatKernelSimilarity:", K_uu.shape)

    print("Class-class similarity graph")
    # Class-class similarity graph #
    # REVIEW: Put outside to the loop if D_s is not updating
    src_sim_src = computeProbability(D_s[0], source_classifiers, normalize_method="sigmoid")
    src_sim_tgt_c = src_sim_src @ GG
    print(src_sim_tgt_c.shape)

    transferred_samples = None
    replace = True      # Starts with replace true to discard randomly chosen 2 samples in labeled set
    print("Let's begin!")
    for it in range(max_iterations):
        print("#%d" % it)
        # Update Heat Kernel similarity matrix #
        if transferred_samples is not None:
            if replace:
                replace = False
                H_st = heatKernelSimilarity_v2(D_s[0], transferred_samples, sigma=sigma)
                H_ts = H_st.T.copy()        # XXX: Not really required
                H_tt = heatKernelSimilarity_v2(L_p[0], transferred_samples, sigma=sigma)
            else:
                H_st = np.hstack((H_st, heatKernelSimilarity_v2(D_s[0], transferred_samples, sigma=sigma)))
                H_ts = H_st.T.copy()        # XXX: Not really required
                H_tts = heatKernelSimilarity_v2(L_p[0], transferred_samples, sigma=sigma)
                H_tt = np.hstack((H_tt, H_tts[:-transferred_samples.shape[0]]))
                H_tt = np.vstack((H_tt, H_tts.T))
            print("HeatKernelSimilarity Updated:", H_ss.shape, H_st.shape, H_ts.shape, H_tt.shape)
        target_indexes = np.arange(L_p[0].shape[0])
        source_indexes = np.arange(D_s[0].shape[0])

        print("Sample-sample similarity graph")
        # Sample-sample similarity graph #
        src_random_samples_idxs = np.random.choice(D_s[0].shape[0], n_random_samples, replace=False)
        H_ss_ = H_ss[np.ix_(src_random_samples_idxs, src_random_samples_idxs)]
        H_st_ = H_st[np.ix_(src_random_samples_idxs, target_indexes)]
        H_ss_, H_st_ = normalize(H_ss_, H_st_, method="l1", return_split=True)
        print(H_ss_.shape, H_st_.shape)

        H_ts_ = H_ts[np.ix_(target_indexes, src_random_samples_idxs)]
        H_tt_ = H_tt.copy()
        Y_tc = np.zeros((L_p[0].shape[0], num_target_classes))      # One hot encoding
        for col, c in enumerate(target_classes):
            Y_tc[L_p[1]==c, col] = 1
        H_ts_, H_tt_, Y_tc = normalize(H_ts_, H_tt_, Y_tc, method="l1", return_split=True)
        print(H_ts_.shape, H_tt_.shape, Y_tc.shape)
        H_st_st = np.vstack((np.hstack((H_ss_, H_st_)), np.hstack((H_ts_, H_tt_))))
        print(H_st_st.shape)
        H_st_c = np.vstack((np.zeros((n_random_samples, num_target_classes)), Y_tc))
        print(H_st_c.shape)
        HH = np.linalg.inv(np.identity(n_random_samples + L_p[0].shape[0]) - H_st_st) @ H_st_c
        print(HH.shape)

        H_xs = H_ss[np.ix_(source_indexes, src_random_samples_idxs)]
        H_xt = H_st.copy()
        H_xs, H_xt = normalize(H_xs, H_xt, method="l1", return_split=True)
        print(H_xs.shape, H_xt.shape)
        src_sim_tgt_s = np.hstack((H_xs, H_xt)) @ HH
        print(src_sim_tgt_s.shape)
        # Combine similarities between source samples to target classes from both graphs #
        src_sim_tgt = lambdaa * src_sim_tgt_c + (1 - lambdaa) * src_sim_tgt_s
        print(src_sim_tgt.shape)

        print("Expanding Labeled Set by adding top related source samples")
        # Expand Labeled Set by adding top related source samples #
        indexes = []
        weights = []
        transfer_labels = []
        for col, c in enumerate(target_classes):
            idx = np.argpartition(src_sim_tgt[:, col], -n_transfer_samples)[-n_transfer_samples:]
            weights += list(src_sim_tgt[:, col][idx])
            indexes += list(idx)
            transfer_labels += [c]*n_transfer_samples
        # indexes = list(set(indexes))        # Remove duplicates if any (set does not preserve order)
        print("[%d/%d] Number of transferred samples: %d" % (it, max_iterations, len(indexes)))
        print(len(list(set(list(indexes)))))
        # NOTE: Allowing duplicate source samples
        expanded_set_L = (np.vstack((L_p[0], D_s[0][indexes])),
                          np.vstack((L_p[1].reshape(-1,1), np.array(transfer_labels).reshape(-1,1))).reshape(-1)
                         )
        L_weights = np.vstack((np.ones(shape=(L_p[1].shape[0], 1)), np.array(weights).reshape(-1,1))).reshape(-1)
        print("Expanded Set L:", expanded_set_L[0].shape, expanded_set_L[1].shape, L_weights.shape)

        print("Constructing classifiers on target classes")
        # Construct classifiers on target classes #
        # target_classifiers = get_classifiers(*expanded_set_L, classifier="svc", kernel="linear",
                                            #    weights=L_weights, random_state=i, ncpu=ncpu)
        target_classifiers = get_classifiers(*expanded_set_L, target_classes, classifier="linearsvc",
                                             weights=L_weights, random_state=i, ncpu=ncpu)
        acc = eval_classifier(target_classifiers, D_t[0], D_t[1], classes=target_classes)
        average_acc[i, it] = acc

        print("Computing Entropy on unlabeled target data")
        # Entropy computation on unlabeled target data #
        U_sim_tgt = computeProbability(U_p[0], target_classifiers, normalize_method="softmax")
        E_u = -np.sum(U_sim_tgt * np.log(U_sim_tgt), axis=1).reshape(-1, 1)
        print(E_u.shape)

        src_rs_idxs = np.random.choice(D_s[0].shape[0], n_random_samples2, replace=False)
        K_us = heatKernelSimilarity_v2(U_p[0], D_s[0][src_rs_idxs], sigma=sigma)
        print("HeatKernelSimilarity of unlabeled data:", K_uu.shape, K_us.shape)

        print("Ranking score of unlabeled samples by solving the convex optimization problem")
        # Ranking score of unlabeled samples by solving the convex optimization problem #
        # NOTE: multiply by 2 as in paper quadratic term is not multiplied by half
        P = cvxopt.matrix((2 * eta * K_uu).astype(np.double))
        q = cvxopt.matrix(-((K_uu @ E_u) + tau*(K_us @ np.ones(shape=(n_random_samples2, 1)))).astype(np.double))
        Gm = cvxopt.matrix((0.0 - np.identity(K_uu.shape[0])).astype(np.double))
        h = cvxopt.matrix(0.0, (K_uu.shape[0], 1))
        A = cvxopt.matrix(1.0, (1, K_uu.shape[0]))
        b = cvxopt.matrix(1.0)
        R_p = np.array(cvxopt.solvers.qp(P, q, Gm, h, A, b)['x']).reshape(-1)
        print("Ranking matrix:", R_p.shape)

        print("Expert Labeling")
        # Expert Labeling #
        u_idx = np.argpartition(R_p, -n_expert_samples)[-n_expert_samples:]
        print("Now let's see the ranking of top %d unlabeled samples:" % n_expert_samples, R_p[u_idx], u_idx)
        transferred_samples = U_p[0][u_idx].copy()
        if replace:
            L_p[0] = transferred_samples.copy()
            L_p[1] = U_p[1][u_idx].copy()
        else:
            L_p[0] = np.vstack((L_p[0], U_p[0][u_idx]))
            L_p[1] = np.vstack((L_p[1].reshape(-1,1), U_p[1][u_idx].reshape(-1,1))).reshape(-1)
        U_p[0], U_p[1] = np.delete(U_p[0], u_idx, axis=0), np.delete(U_p[1], u_idx, axis=0)
        print("Updated labeled and unlabeled data:")
        print(L_p[0].shape, L_p[1].shape, U_p[0].shape, U_p[1].shape)
        K_uu = np.delete(np.delete(K_uu, u_idx, axis=0), u_idx, axis=1)
        print("Updated K_uu:", K_uu.shape)
        print("Iteration #%d completed!" % it)

average_acc = np.average(average_acc, axis=0).reshape(-1,1)
overall_acc = np.ones(shape=(max_iterations, 1)) * np.average(overall_acc)
average_acc_plot = Plotter("plots/cifar10_%s_atl.jpeg" % args.model, num_lines=2, legends=["All samples", "ATL with Cross-class similarity transfer"],
                            xlabel="Number of iterations", ylabel="Accuracy (%)", title="Accuracy vs Iterations" )
iters = np.arange(max_iterations).reshape(-1,1)
average_acc_plot(np.hstack((iters, overall_acc)), np.hstack((iters, average_acc)))
# average_acc_plot.queue.put(None)
average_acc_plot.queue.join()
average_acc_plot.clean_up()
