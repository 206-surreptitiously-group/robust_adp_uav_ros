import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# dx = 2x + u
if __name__ == '__main__':
    r = pd.read_csv('test_record.csv')
    plt.plot(r['reward'])
    plt.title('reward')
    plt.show()
