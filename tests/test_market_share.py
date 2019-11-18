import numpy as np

# 2 random coefficients
# 10 products

# π - K x D
# Σ - K x K
#
# delta - J x 1
# x - J x K
# D - D x I
# v - K x I


K = 2
D = 3
J = 10
I = 20
ns = I

pi = np.random.normal(size=(K, D))
sigma = np.random.normal(size=(K, K))

delta = np.random.normal(size=J)
x = np.random.normal(size=(J, K))
d = np.random.normal(size=(D, I))
v = np.random.normal(size=(K, I))

exp_utility = np.exp(delta[:, np.newaxis] + (x @ ((pi @ d) + (sigma @ v))))
np.sum(exp_utility / (1 + np.sum(exp_utility, axis=0)), axis=1)


def test_hello():
    pass