# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model, get_err
import math
import sys
import operator
		
def main():
    movie_info = np.genfromtxt('../155project2/data/movies.txt', dtype='str', delimiter="\t", usecols=(0, 1, 3, 7, 16))
    data = np.loadtxt('../155project2/data/data.txt').astype(int)
    Y_train = np.loadtxt('../155project2/data/train.txt').astype(int)
    Y_test = np.loadtxt('../155project2/data/test.txt').astype(int)

	
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
	
    reg = 0.0
    eta = 0.03 # learning rate
    E_in = []
    E_out = []

    # Use to compute Ein and Eout

    U,V, err = train_model(M, N, 20, eta, reg, Y_train)
	
    a, sigma, b = np.linalg.svd(V)
    print(V.shape, a.shape)
    a_t = np.transpose(a)
    v_proj = np.dot(a_t[:2], V)

    movie_ids = v_proj[0]
    evals = v_proj[1]

    ratings = {}

    for i in range(len(movie_ids)):
        movie_id = movie_ids[i]
        rating = evals[i]
        if movie_id in ratings:
            ratings[movie_id].append(rating)
        else:
            ratings[movie_id] = [rating]


    # 1. 10 movies of our choice from the MovieLens dataset 


    # 2. All ratings of the ten most popular movies 

    max_10 = dict(sorted(ratings.items(), key=lambda r: len(r[1]), reverse=True)[:10])
    top_ratings = []
    [top_ratings.extend(v) for v in max_10.values()]
    plt.hist(top_ratings, bins=5)
    plt.xlabel('ratings')
    plt.ylabel('frequency')
    plt.title('Histogram of all ratings of the ten most popular movies')
    plt.savefig('popular_51b.png')
    plt.clf()


    # 3. All ratings of the ten best movies 

    best_10 = dict(sorted(ratings.items(), key=lambda r: sum(r[1])/len(r[1]), reverse=True)[:10])
    best_ratings = []
    [best_ratings.extend(v) for v in best_10.values()]
    plt.hist(best_ratings, bins=5)
    plt.xlabel('ratings')
    plt.ylabel('frequency')
    plt.title('Histogram of all ratings of the ten best movies')
    plt.savefig('best_51c.png')
    plt.clf()


    # 4. All ratings of movies from three genres of your choice

    ids = movie_info[:,0].astype(int)
    movie_names = movie_info[:,1]

    # Action:

    action = movie_info[:,2].astype(int)
    action_movies = dict((k, v) for k, v in zip(ids, action) if v == 1)
    action_ratings_dict = dict((k, ratings[k]) for k in action_movies.keys())
    action_ratings = []
    [action_ratings.extend(v) for v in action_ratings_dict.values()]
    plt.hist(action_ratings, bins=5)
    plt.xlabel('ratings')
    plt.ylabel('frequency')
    plt.title('Histogram of all ratings of action movies')
    plt.savefig('action_51di.png')
    plt.clf()

    # Comedy:

    comedy = movie_info[:,3].astype(int)
    comedy_movies = dict((k, v) for k, v in zip(ids, comedy) if v == 1)
    comedy_ratings_dict = dict((k, ratings[k]) for k in comedy_movies.keys())
    comedy_ratings = []
    [comedy_ratings.extend(v) for v in comedy_ratings_dict.values()]
    plt.hist(comedy_ratings, bins=5)
    plt.xlabel('ratings')
    plt.ylabel('frequency')
    plt.title('Histogram of all ratings of comedies')
    plt.savefig('comedy_51dii.png')
    plt.clf()

    # Romance:

    romance = movie_info[:,4].astype(int)
    romance_movies = dict((k, v) for k, v in zip(ids, romance) if v == 1)
    romance_ratings_dict = dict((k, ratings[k]) for k in romance_movies.keys())
    romance_ratings = []
    [romance_ratings.extend(v) for v in romance_ratings_dict.values()]
    plt.hist(romance_ratings, bins=5)
    plt.xlabel('ratings')
    plt.ylabel('frequency')
    plt.title('Histogram of all ratings of romance movies')
    plt.savefig('romance_51diii.png')
    plt.clf()




   

if __name__ == "__main__":
    main()
