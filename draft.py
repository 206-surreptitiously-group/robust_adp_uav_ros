import numpy as np
import matplotlib.pyplot as plt


# dx = 2x + u
if __name__ == '__main__':
    pos_zone = np.atleast_2d([[-1, 0], [0, 1], [1, 2]])
    print(np.random.uniform(low=pos_zone[:, 0] + 0.3, high=pos_zone[:, 1] - 0.3))
