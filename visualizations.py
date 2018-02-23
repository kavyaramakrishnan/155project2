# BASIC VISUALIZATIONS

import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import operator

# movies = np.genfromtxt('../project2/data/movies.txt', dtype='str')
movie_info = np.genfromtxt('../data/movies.txt', dtype='str', delimiter="\t", usecols=(0, 1, 3, 7, 16))
data = np.loadtxt('../data/data.txt').astype(int)
train = np.loadtxt('../data/train.txt').astype(int)
test = np.loadtxt('../data/test.txt').astype(int)

ratings = {}
for user, movie_id, rating in data:
    if movie_id in ratings:
        ratings[movie_id].append(rating)
    else:
        ratings[movie_id] = [rating]


# 1. All ratings in the MovieLens Dataset

all_ratings = data[:,2]
plt.hist(all_ratings, bins=5)
plt.xlabel('ratings')
plt.ylabel('frequency')
plt.title('Histogram of all ratings in the MovieLens Dataset')
plt.savefig('all.png')
plt.clf()


# 2. All ratings of the ten most popular movies 

max_10 = dict(sorted(ratings.iteritems(), key=lambda r: len(r[1]), reverse=True)[:10])
top_ratings = []
[top_ratings.extend(v) for v in max_10.values()]
plt.hist(top_ratings, bins=5)
plt.xlabel('ratings')
plt.ylabel('frequency')
plt.title('Histogram of all ratings of the ten most popular movies')
plt.savefig('popular.png')
plt.clf()


# 3. All ratings of the ten best movies 

best_10 = dict(sorted(ratings.iteritems(), key=lambda r: sum(r[1])/len(r[1]), reverse=True)[:10])
best_ratings = []
[best_ratings.extend(v) for v in best_10.values()]
plt.hist(best_ratings, bins=5)
plt.xlabel('ratings')
plt.ylabel('frequency')
plt.title('Histogram of all ratings of the ten best movies')
plt.savefig('best.png')
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
plt.savefig('action.png')
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
plt.savefig('comedy.png')
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
plt.savefig('romance.png')
plt.clf()
