# Recommender-System

This is the code for the Recommender System which uses the lightfm recommender system library to train a hybrid algorithm that uses the WARP loss function on the movielens dataset. 
It prints out recommended movies for whatever user id from the dataset that we choose.

The WARP loss function is used here after plotting and comparing the AUC score of the 4 loss functions present in lightfm library-
1. WARP (Weighted Approximate-Rank Pairwise)
2. BRP (Bayesian Personalised Ranking)
3. k-OS WARP 
4. Logistic


##Dependencies

1. numpy (http://www.numpy.org/)
2. scipy (https://www.scipy.org/)
3. lightfm (https://github.com/lyst/lightfm)
4. matplotlib (https://matplotlib.org/)
