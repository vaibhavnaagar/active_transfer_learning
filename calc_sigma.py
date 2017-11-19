import numpy as np
import pickle
import argparse
from sklearn.metrics.pairwise import euclidean_distances

def compute_sigma(datafile):
    with open(datafile, 'rb') as f:
        data = pickle.load(f)
        train_data = data['train_features']
        train_labels = data['train_labels'].reshape(-1)
        test_data = data['test_features']
        test_labels = data['test_labels'].reshape(-1)
    return np.average(euclidean_distances(train_data, train_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active Transfer Learning with Cross-class Similarity Transfer')
    parser.add_argument('--dset', '-d', required=True, help='Path to dataset')
    args = parser.parse_args()
    sigma = compute_sigma(args.dset)
    print(sigma)

# 15.0467578136 (512 preactnet)
# 60.608 (4096 alexnet)
# 21.8557 (512 resnet18)
