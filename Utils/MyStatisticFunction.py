# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    MyStatisticFunction.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gigregoi <gigregoi@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/09/16 19:15:14 by gigregoi          #+#    #+#              #
#    Updated: 2020/12/24 07:05:12 by gigregoi         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def count_(X):
    try:
        X = X.astype('float')
        X = X[~np.isnan(X)]
        return len(X)
    except:
        return len(X)

def mean_(X):
    total = 0
    for x in X:
        if np.isnan(x):
            continue
        total = total + x
    return total / len(X)

def std_(X):
    mean = mean_(X)
    total = 0
    # X = X[~np.isnan(X)]
    for x in X:
        if np.isnan(x):
            continue
        total = total + (x - mean) ** 2
    return (total / (count_(X)-1)) ** 0.5

def min_(X):
    value = X[0]
    for x in X:
        val = x
        if val < value:
            value = val
    return value

def max_(X):
    value = X[0]
    for x in X:
        val = x
        if val > value:
            value = val
    return value

def percentile_(X, p):
    X.sort()
    k = (len(X)-1) * (p / 100) # k n'est pas forcement un entier
    f = np.floor(k)
    c = np.ceil(k)

    if f == c:
        return X[int(k)]

    d0 = X[int(f)] * (c - k) # on affecte un poids equivalent a la dstance entre k et l'entier sup c
    d1 = X[int(c)] * (k - f) # de meme avec l'entier inf
    return d0 + d1
