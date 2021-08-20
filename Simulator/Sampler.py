from random import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from scipy.stats import beta, poisson

class BaseSampler():
    def __init__(self, random_seed=None) -> None:
        self.random_seed = random_seed
        self.rv = None
        if random_seed is not None:
            np.random.seed(random_seed)
            self.rv_state = np.random.RandomState(seed=random_seed)

    def sample(self, n=1):
        assert(n > 0)
        if n == 1:
            return self.rv.rvs(1)[0]
        else:
            return self.rv.rvs(size=n)
    
    def plot(self):
        raise NotImplementedError

class IdentitySampler(BaseSampler):

    def __init__(self, value) -> None:
        super().__init__(random_seed=None)
        self.value = value
    
    def sample(self, n=1):
        assert(n > 0)
        if n == 1:
            return self.value
        else:
            return np.full((n,), self.value)

class BetaSampler(BaseSampler):

    def __init__(self, a=9, b=2, random_seed=None) -> None:
        super().__init__(random_seed=random_seed)

        self.rv = beta(a, b)
        if self.random_seed is not None:
            self.rv.random_state = self.rv_state
    
    def plot(self):
        x = np.arange(0.01, 1, 0.01)
        y = self.rv.pdf(x)
        plt.plot(x,y)
        plt.show()

class PoissonSampler(BaseSampler):

    def __init__(self, mu=5., random_seed=None) -> None:
        super().__init__(random_seed=random_seed)
        self.mu = mu
        self.rv = poisson(mu)
        if self.random_seed is not None:
            self.rv.random_state = self.rv_state
    
    def plot(self):
        x = np.arange(poisson.ppf(0.01, self.mu), poisson.ppf(0.99, self.mu))
        _, ax = plt.subplots(1, 1)
        ax.plot(x, poisson.pmf(x, self.mu), 'bo', ms=8, label='poisson pmf')
        ax.vlines(x, 0, poisson.pmf(x, self.mu), colors='b', lw=5, alpha=0.5)
        plt.show()
        

if __name__ == '__main__':
    # a = Sampler(random_seed=20200430)
    # print(a.beta(10, debug=False))
    sampler = BetaSampler(random_seed=1)
    print(sampler.plot())
