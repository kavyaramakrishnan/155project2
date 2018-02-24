# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model_bias, get_err_bias
import math
import sys
import operator
from adjustText import adjust_text
        
def main():
    #movie_info = np.genfromtxt('../data/movies.txt', dtype="str", delimiter="\t", usecols=(0, 1, 3, 7, 16))
    movie_info = np.loadtxt('../data/movies.txt', dtype="str", delimiter="\t", usecols=(0, 1, 3, 7, 16))
    data = np.loadtxt('../data/data.txt').astype(int)
    Y_train = np.loadtxt('../data/train.txt').astype(int)
    Y_test = np.loadtxt('../data/test.txt').astype(int)
    print(movie_info)
    
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    
    reg = 0.0
    eta = 0.03 # learning rate
    k = 20
    E_in = []
    E_out = []

    # Use to compute Ein and Eout

    U, V, UBias, VBias, err = train_model_bias(M, N, k, eta, reg, Y_train)
    e_out = get_err_bias(U, V, UBias, VBias, Y_test)
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


    # Setup

    ids = movie_info[:,0].astype(int)
    movie_names = movie_info[:,1]



    # 1. 10 movies of our choice from the MovieLens dataset 
    
    plt.scatter(x[2:12], y[2:12])
    texts = []
    for j, txt in enumerate(movie_names[2:12]):
        texts.append(plt.text(x[2:12][j], y[2:12][j], txt))
    adjust_text(texts)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('10 Movies of Our Choice')
    plt.savefig('Bias-choice.png')
    plt.clf()

    # 2. All ratings of the ten most popular movies 

    max_10 = dict(sorted(ratings.items(), key=lambda r: len(r[1]), reverse=True)[:10])
    x_pop = []
    y_pop = []
    top_ratings = []
    top_ratings = max_10.keys()
    movie_title = []
    print(top_ratings)
    counter = 0
    for i in v_proj:
        counter += 1
        if counter in top_ratings:
            x_pop.append(i[0])
            y_pop.append(i[1])
            movie_title.append(movie_names[counter])
    print(movie_title)

    plt.scatter(x_pop, y_pop)
    texts = []
    for j, txt in enumerate(movie_title):
        texts.append(plt.text(x_pop[j], y_pop[j], txt))
        #plt.annotate(txt, (x_pop[j], y_pop[j]))
    adjust_text(texts)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('10 Most Popular Movies')
    plt.savefig('Bias-popular.png')
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

    for j, txt in enumerate(movie_title):
        plt.annotate(txt, (x_best[j], y_best[j]))
    plt.scatter(x_best, y_best)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('10 Best Movies')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig('Bias-best.png')
    plt.clf()


    # 4. 10 ratings of movies from three genres of your choice


    ids = movie_info[:,0].astype(int)
    movie_names = movie_info[:,1]


    action = (movie_info[:,2].astype(int))
    action_movies = dict((k, v) for k, v in zip(ids, action) if v == 1)
    action_ratings_dict = dict((k, ratings[k]) for k in action_movies.keys())
    x_action = []
    y_action = []
    action_ratings = []
    action_ratings = action_ratings_dict.keys()

    comedy = movie_info[:,3].astype(int)
    comedy_movies = dict((k, v) for k, v in zip(ids, comedy) if v == 1)
    comedy_ratings_dict = dict((k, ratings[k]) for k in comedy_movies.keys())
    x_comedy = []
    y_comedy = []
    comedy_ratings = []
    comedy_ratings = comedy_ratings_dict.keys()

    romance = movie_info[:,4].astype(int)
    romance_movies = dict((k, v) for k, v in zip(ids, romance) if v == 1)
    romance_ratings_dict = dict((k, ratings[k]) for k in romance_movies.keys())
    x_romance = []
    y_romance = []
    romance_ratings = []
    romance_ratings = romance_ratings_dict.keys()

    count = 0
    for i in v_proj:
        count += 1
        if count in action_ratings:
            x_action.append(i[0])
            y_action.append(i[1])

        if count in comedy_ratings:
            x_comedy.append(i[0])
            y_comedy.append(i[1])

        if count in romance_ratings:
            x_romance.append(i[0])
            y_romance.append(i[1])

    plt.scatter(x_action[2:12], y_action[2:12], label = "Action")
    plt.scatter(x_comedy[2:12], y_comedy[2:12], color = 'orange', label = "Comedy")
    plt.scatter(x_romance[2:12], y_romance[2:12], color = 'green', label = "Romance")
    plt.legend()
    plt.title("Three Genres")
    plt.savefig('Bias-genres.png')
    plt.clf()
    

if __name__ == "__main__":
    main()
