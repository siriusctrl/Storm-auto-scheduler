from random import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from scipy.stats import beta, poisson

class BaseSampler():
    def __init__(self, random_seed=None) -> None:
        self.random_seed = random_seed

        if random_seed:
            np.random.seed(random_seed)
            self.rv_state = np.random.RandomState(seed=random_seed)

    def sample(self, n):
        raise NotImplementedError

class IdentitySampler(BaseSampler):

    def __init__(self, value, random_seed=None) -> None:
        super().__init__(random_seed=random_seed)
        self.value = value
    
    def sample(self, n=1):
        assert(n > 0)
        if n == 1:
            return self.value
        else:
            return np.full((n,), self.value)

class BetaSampler(BaseSampler):

    def __init__(self, random_seed=None) -> None:
        super().__init__(random_seed=random_seed)

    def sample(self, n=1, a=9, b=2, plot=False):
        assert(n > 0)
        rv = beta(a, b)
        if self.random_seed is not None:
            rv.random_state = self.rv_state

        if plot:
            x = np.arange(0.01, 1, 0.01)
            y = rv.pdf(x)
            plt.plot(x,y)
            plt.show()

        return rv.rvs(size=n)

class PoissonSampler(BaseSampler):

    def __init__(self, random_seed) -> None:
        super().__init__(random_seed=random_seed)

    def sample(self, n=1, mu=5., plot=False):
        assert(n > 0)
        rv = poisson(mu)
        if self.random_seed is not None:
            rv.random_state = self.rv_state
        
        if plot:
            x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
            _, ax = plt.subplots(1, 1)
            ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
            ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
            plt.show()

        return rv.rvs(size=n)
        

if __name__ == '__main__':
    # a = Sampler(random_seed=20200430)
    # print(a.beta(10, debug=False))
    sampler = PoissonSampler(random_seed=1)
    print(sampler.sample(n=10, mu=20., plot=True))
    print(sampler.sample(n=10))
