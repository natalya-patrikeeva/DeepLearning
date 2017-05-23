"""Softmax. """
import numpy as np

# A classifier outputs three scores for three classes.
scores = [3.0, 1.0, 0.2] # a single sample
scores = np.array(scores)
# Test one dimensional input
# scores = [1.0, 2.0, 3.0]

# two dimensional input
# scores = np.array([[1, 2, 3, 6],
#                    [2, 4, 5, 6],
#                    [3, 8, 7, 6]])

#
def softmax(x):
    """Compute softmax values for each sets of scores in x.
    x is numpy array for one row for each score and arbitrary number
    of columns, one for each sample"""

    # rows are scores
    # columns are samples
    return np.exp(x) / np.sum( np.exp(x), axis=0 )


print(softmax(scores))
print(softmax(scores * 10)) # as the scores increase, they get close to 0 or 1
print(softmax(scores / 10))  # as the scores decrease, they get closer to uniform

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)

scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2 )
plt.legend(('x', '1', '0.2'), loc="upper right")
plt.show()
