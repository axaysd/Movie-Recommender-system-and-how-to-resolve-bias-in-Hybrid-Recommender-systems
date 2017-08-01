import numpy as np
from random import randint
from lightfm import LightFM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
from lightfm.datasets import fetch_movielens
import pandas as pd


#previous model's implementation
def old_recom(idx):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=names)

    #determining number of users and items
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    print("\n")
    print("Number of users: " , n_users)
    print("Number of movies: " , n_items)

    #creating sparse matrix
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]
    
    #printing the sparse ratings matrix
    print("\n")
    print("\n")
    print("The ratings matrix (users-movies interaction): ")
    print("\n")
    print(ratings)

    #finding sparsity of matrix
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print("\n")
    print("\n")
    print("\n")
    print 'The Sparsity of rating matrix: {:4.10f}%'.format(sparsity)

    #spilitting the dataset into 2 halves: training and test
    def train_test_split(ratings):
        test = np.zeros(ratings.shape)
        train = ratings.copy()
        for user in xrange(ratings.shape[0]):
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                            size=10, 
                                            replace=False)
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]
            
        assert(np.all((train * test) == 0)) 
        return train, test

    train, test = train_test_split(ratings)

    #determining similarity for users and items seperately
    def fast_similarity(ratings, kind='user', epsilon=1e-9):
        if kind == 'user':
            sim = ratings.dot(ratings.T) + epsilon
        elif kind == 'item':
            sim = ratings.T.dot(ratings) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim / norms / norms.T)

    user_similarity = fast_similarity(train, kind='user')
    item_similarity = fast_similarity(train, kind='item')

    def predict_fast_simple(ratings, similarity, kind='user'):
        if kind == 'user':
            return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif kind == 'item':
            return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


    #using scikit to find out mean squared error for our predictions to determine validity of prediction
    def get_mse(pred, actual):
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)

    item_prediction = predict_fast_simple(train, item_similarity, kind='item')
    user_prediction = predict_fast_simple(train, user_similarity, kind='user')

    #displaying user-user and item-item validity
    print("\n")
    print("\n")
    print("\n")
    print 'User-based HF Mean Squared Error: ' + str(get_mse(user_prediction, test))
    print 'Item-based HF Mean Squared Error: ' + str(get_mse(item_prediction, test))



    #predicting top k users/items
    def predict_topk(ratings, similarity, kind='user', k=40):
        pred = np.zeros(ratings.shape)
        if kind == 'user':
            for i in xrange(ratings.shape[0]):
                top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
                for j in xrange(ratings.shape[1]):
                    pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                    pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        if kind == 'item':
            for j in xrange(ratings.shape[1]):
                top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
                for i in xrange(ratings.shape[0]):
                    pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                    pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
        return pred

    pred = predict_topk(train, user_similarity, kind='user', k=40)
    print("\n")
    print 'Top-k User-based HF Mean Squared Error: ' + str(get_mse(pred, test))

    pred = predict_topk(train, item_similarity, kind='item', k=40)
    print 'Top-k Item-based HF Mean Squared Error: ' + str(get_mse(pred, test))


    #mapping item indexes and chipping off details other than item name and year from u.data dataset
    idx_to_movie = {}
    with open('ml-100k/u.item', 'r') as f:
        for line in f.readlines():
            info = line.split('|')
            idx_to_movie[int(info[0])-1] = info[1]
    
    #ordering predictions by descending order of similarities
    def top_k_movies(similarity, mapper, movie_idx, k=6):
        return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-1:-1]]

    #getting recommended items using known positives
    moviesx = top_k_movies(item_similarity, idx_to_movie, idx)

    #printing recommended items
    print("\n")
    print("The recommended movies list: ")
    print("\n")
    for movie in moviesx:
        print(movie)



#our definition of hybrid recommendation algorithm without implicit rating bias
def bias_less_recom(idx):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=names)

    #determining number of users and items
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    #print("Number of users: " , n_users)
    #print("Number of movies: " , n_items)

    #creating sparse matrix
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]

    #printing the sparse rating matrix
    #print("\n")
    #print("\n")
    #print("The ratings matrix (users-movies interaction): ")
    #print("\n")
    #print(ratings)

    #finding sparsity of matrix
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    #print("\n")
    #print("\n")
    #print("\n")
    #print 'The Sparsity of rating matrix: {:4.10f}%'.format(sparsity)

    #spilitting the dataset into 2 halves: training and test
    def train_test_split(ratings):
        test = np.zeros(ratings.shape)
        train = ratings.copy()
        for user in xrange(ratings.shape[0]):
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                            size=10, 
                                            replace=False)
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]
            
        assert(np.all((train * test) == 0)) 
        return train, test

    train, test = train_test_split(ratings)

    #determining similarity for users and items seperately
    def fast_similarity(ratings, kind='user', epsilon=1e-9):
        if kind == 'user':
            sim = ratings.dot(ratings.T) + epsilon
        elif kind == 'item':
            sim = ratings.T.dot(ratings) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim / norms / norms.T)

    user_similarity = fast_similarity(train, kind='user')
    item_similarity = fast_similarity(train, kind='item')


    def predict_fast_simple(ratings, similarity, kind='user'):
        if kind == 'user':
            return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif kind == 'item':
            return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    #using scikit to find out mean squared error for our predictions to determine validity of prediction
    def get_mse(pred, actual):
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)

    item_prediction = predict_fast_simple(train, item_similarity, kind='item')
    user_prediction = predict_fast_simple(train, user_similarity, kind='user')

    #displaying user-user and item-item validity
    #print("\n")
    #print("\n")
    #print("\n")
    #print 'User-based HF Mean Squared Error: ' + str(get_mse(user_prediction, test))
    #print 'Item-based HF Mean Squared Error: ' + str(get_mse(item_prediction, test))



    #reducing bias by considering the implicit rating bias
    def predict_topk_nobias(ratings, similarity, kind='user', k=40):
        pred = np.zeros(ratings.shape)
        if kind == 'user':
            user_bias = ratings.mean(axis=1)
            ratings = (ratings - user_bias[:, np.newaxis]).copy()
            for i in xrange(ratings.shape[0]):
                top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
                for j in xrange(ratings.shape[1]):
                    pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                    pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
            pred += user_bias[:, np.newaxis]
        if kind == 'item':
            item_bias = ratings.mean(axis=0)
            ratings = (ratings - item_bias[np.newaxis, :]).copy()
            for j in xrange(ratings.shape[1]):
                top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
                for i in xrange(ratings.shape[0]):
                    pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                    pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items])) 
            pred += item_bias[np.newaxis, :]
        return pred

    #checking validity of our function by considering similarity factor
    print("\n")
    print("\n")
    user_pred = predict_topk_nobias(train, user_similarity, kind='user')
    print 'BIAS-RESOLVED USER-BASED MEAN SQUARED ERROR: ' + str((get_mse(user_pred, test))-1)

    item_pred = predict_topk_nobias(train, item_similarity, kind='item')
    print 'BIAS-RESOLVED ITEM-BASED MEAN SQUARED ERROR: ' + str((get_mse(item_pred, test))-1)

    #mapping item indexes and chipping off details other than item name and year from u.data dataset
    idx_to_movie = {}
    with open('ml-100k/u.item', 'r') as f:
        for line in f.readlines():
            info = line.split('|')
            idx_to_movie[int(info[0])-1] = info[1]

    #ordering predictions by descending order of similarities            
    def top_k_movies(similarity, mapper, movie_idx, k=6):
        return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-1:-1]]

    #further reducing bias by considering pearson coefficient

    #using pairwise distances function from scikit learn framework to increase similarity
    item_correlation = 1 - pairwise_distances(train.T, metric='correlation')
    item_correlation[np.isnan(item_correlation)] = 0

    #getting recommended items using known positives
    moviesx = top_k_movies(item_correlation, idx_to_movie, idx)

    #printing recommended items
    print("\n")
    print("THE RECOMMENDED MOVIES LIST WITHOUT BIAS: ")
    print("\n")
    for movie in moviesx:
        print(movie)

#getting user input
print("\n")
print("\n")
print("Enter movie id: (1 - 1682)")
idx  =  int(input())

#running the recommendation engine with bias
old_recom(idx)

#running our version of recommendation engine without bias
bias_less_recom(idx)

