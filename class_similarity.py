import gensim
import numpy as np
import pickle

def compute_similarity(words):
	binary_file = "./data/GoogleNews-vectors-negative300.bin"
	word_vectors = gensim.models.KeyedVectors.load_word2vec_format(binary_file, binary=True)
	sims =  np.array([[word_vectors.similarity(w1, w2) for w1 in words] for w2 in words])
	del word_vectors
	return sims

if __name__ == '__main__':
	classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	G = compute_similarity(classes)
	print(G)
	print(G.shape)
	with open('cifar10_class_similarity_matrix_G.pkl', 'wb') as f:
            pickle.dump(G, f)
