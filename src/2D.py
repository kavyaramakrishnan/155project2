# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model, get_err
import math
import sys
import operator
        
def main():
    movie_info = np.genfromtxt('../data/movies.txt', dtype='str', delimiter="\t", usecols=(0, 1, 3, 7, 16))
    data = np.loadtxt('../data/data.txt').astype(int)
    Y_train = np.loadtxt('../data/train.txt').astype(int)
    Y_test = np.loadtxt('../data/test.txt').astype(int)

    
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    
    reg = 0.2
    eta = 0.03 # learning rate
    E_in = []
    E_out = []

    # Use to compute Ein and Eout

    U,V, err = train_model(M, N, 20, eta, reg, Y_train)
    e_out = get_err(U, V, Y_test)
    print("e_in", err)
    print("e_out", e_out)

    #model.score(Y_test)
    a, sigma, b = np.linalg.svd(V)
    print(V.shape, a.shape)
    a_t =  a #np.transpose(a)

    #movie ID starts at 1, but matrix starts at 0
    v_proj = np.transpose(np.dot(a_t[:2], V))

    x = []
    y = []
    for i in v_proj:
        x.append(i[0])
        y.append(i[1])

    ratings = {}
    for user, movie_id, rating in data:
        if movie_id in ratings:
            ratings[movie_id].append(rating)
        else:
            ratings[movie_id] = [rating]
    #x = v_proj[0]
    #y = v_proj[1]
    #print(x)

    print(v_proj.shape)

    # 1. 10 movies of our choice from the MovieLens dataset 

    plt.scatter(x[2:12], y[2:12])
    plt.savefig('choice.png')
    plt.clf()

    # 2. All ratings of the ten most popular movies 

    max_10 = dict(sorted(ratings.items(), key=lambda r: len(r[1]), reverse=True)[:10])
    x_pop = []
    y_pop = []
    top_ratings = []
    top_ratings = max_10.keys()
    print(top_ratings)
    counter = 0
    for i in v_proj:
        counter += 1
        if counter in top_ratings:
            x_pop.append(i[0])
            y_pop.append(i[1])

    plt.scatter(x_pop, y_pop)
    plt.savefig('popular.png')
    plt.clf()


    # 3. All ratings of the ten best movies 

    best_10 = dict(sorted(ratings.items(), key=lambda r: sum(r[1])/len(r[1]), reverse=True)[:10])
    x_best = []
    y_best = []
    best = []
    best = best_10.keys()
    print(best)
    count = 0
    for i in v_proj:
        count += 1
        if count in best:
            x_best.append(i[0])
            y_best.append(i[1])

    plt.scatter(x_best, y_best)
    plt.savefig('best.png')
    plt.clf()


    # 4. All ratings of movies from three genres of your choice


    ids = movie_info[:,0].astype(int)
    movie_names = movie_info[:,1]

    # Action:

    action = (movie_info[:,2].astype(int))[:90000]
    action_movies = dict((k, v) for k, v in zip(ids, action) if v == 1)
    action_ratings_dict = dict((k, ratings[k]) for k in action_movies.keys())
    x_best = []
    y_best = []
    action_ratings = []
    action_ratings = action_ratings.keys()
    count = 0
    for i in v_proj:
        count += 1
        if count in action_ratings:
            x_best.append(i[0])
            y_best.append(i[1])

    plt.scatter(x_best, y_best)
    plt.savefig('action.png')
    plt.clf()
    

    # # Comedy:

    # comedy = movie_info[:,3].astype(int)
    # comedy_movies = dict((k, v) for k, v in zip(ids, comedy) if v == 1)
    # comedy_ratings_dict = dict((k, ratings[k]) for k in comedy_movies.keys())
    # comedy_ratings = []
    # [comedy_ratings.extend(v) for v in comedy_ratings_dict.values()]
    # plt.hist(comedy_ratings, bins=5)
    # plt.xlabel('ratings')
    # plt.ylabel('frequency')
    # plt.title('Histogram of all ratings of comedies')
    # plt.savefig('comedy_51dii.png')
    # plt.clf()

    # # Romance:

    # romance = movie_info[:,4].astype(int)
    # romance_movies = dict((k, v) for k, v in zip(ids, romance) if v == 1)
    # romance_ratings_dict = dict((k, ratings[k]) for k in romance_movies.keys())
    # romance_ratings = []
    # [romance_ratings.extend(v) for v in romance_ratings_dict.values()]
    # plt.hist(romance_ratings, bins=5)
    # plt.xlabel('ratings')
    # plt.ylabel('frequency')
    # plt.title('Histogram of all ratings of romance movies')
    # plt.savefig('romance_51diii.png')
    # plt.clf()




   

if __name__ == "__main__":
    main()
