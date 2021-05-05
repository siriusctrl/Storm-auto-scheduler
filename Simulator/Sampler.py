import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm

class Sampler():

    def __init__(self, random_seed=None):
        self.random_seed = random_seed

        if self.random_seed is not None:
            self.rv_state = np.random.RandomState(seed=random_seed)
        
        self.b = None
        self.norm = None

    
    def beta(self, n, a=9, b=2, debug=False):
        if self.b is None:
            rv = beta(a, b)
            if self.random_seed is not None:
                rv.random_state = self.rv_state

        if debug:
            x = np.arange(0.01, 1, 0.01)
            y = rv.pdf(x)
            plt.plot(x,y)
            plt.show()

        return rv.rvs(size=n)

    # def normal(self, n, mu=0, sigma=1, debug=False): 
    #     rv = norm(mu, sigma)

    #     if debug:
    #         x = np.arange(-1, 1, 0.01)
    #         y = rv.pdf(x)
    #         plt.plot(x,y)
    #         plt.show()

    #     return rv.rvs(size=n)

if __name__ == '__main__':
    a = Sampler(random_seed=20200430)
    print(a.beta(10, debug=False))