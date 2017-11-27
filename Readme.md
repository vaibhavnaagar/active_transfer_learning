# Active Transfer Learning #
> CS772 (Probabilistic Machine Learning)

* Active Learning with Cross-Class Similarity Transfer -
 [Guo, Yuchen, et al. "Active Learning with Cross-Class Similarity Transfer." AAAI. 2017.](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14165)

* `pretrained_model.py`: Run a specified pretrained model on a specified dataset and generate feature vectors of dataset and store in a **_pickle_** file as `<dataset>_<model>_features.pkl`
```shell
python pretrained_model.py --dataset <dataset_name> --model <model_name> --cuda --ngpu <int>
```
  Use `-h` option for more help.

* `class_similarity.py`: Compute similarity between classes using Word2Vec model trained on **GoogleNews-vectors** and store in a **_pickle_** file. Requires [`GoogleNews-vectors-negative300.bin`](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) in the same directory and specify classes inside the code
```shell
python compute_similarity.py
```

* `active_transfer_learning_parallel.py`: Run the Active Transfer Learning algorithm parallely on specified number of CPU cores. Set hyper-paramters manually in the code. Generate plots inside the `plots` folder with name as `<dset>_<model_name>_atl.jpeg`. Requires dataset file (feature vectors in **pickle** file) and class similarity matrix.  
```
python active_transfer_learning_parallel.py
          -d, --dset <Path to dataset> (required)
          -g, --G <Path to class similarity matrix> (required)
          -m, --model <Model used to construct feature vectors> (default=alexnet)
          -l, --nlabels <Number of labels or classes in dataset> (default=10)
          -w, --workers <Number of CPU cores> (default=1)
          -s, --sigma <Sigma for heat kernel similarity> (default=0.0 and if not specified then will be calculated by code itself)
```
Use `-h` option for more help.

## Requirements
* PyTorch
* Gensim
* Python-3.6 with packages: numpy, cvxopt, gensim
