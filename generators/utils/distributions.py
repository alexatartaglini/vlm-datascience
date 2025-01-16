import numpy as np


class Distribution:
    """
    Base class for probability distributions that can generate random samples.
    """
    def __init__(self, seed=None, dtype=float):
        self.seed = seed
        self.dtype = dtype
        if seed is not None:
            np.random.seed(seed)
            
    def __call__(self, n, **kwargs):
        """Generate n random samples from the distribution"""
        raise NotImplementedError
        
    def __str__(self):
        raise NotImplementedError

    def sample(self, n, **kwargs):
        """Generate n random samples from the distribution"""
        return self(n, **kwargs)


class NormalDistribution(Distribution):
    """
    Normal (Gaussian) distribution with given mean and standard deviation. 
    """
    def __init__(self, mean=0, std=1, seed=None, dtype=float):
        super().__init__(seed, dtype)
        self.type = "numeric"
        self.mean = mean
        self.std = std
        
    def __call__(self, n, **kwargs):
        """Generate n samples from normal distribution"""
        params = {
            'mean': kwargs.get('mean', self.mean),
            'std': kwargs.get('std', self.std),
        }
        return np.random.normal(params['mean'], params['std'], size=n).astype(self.dtype)
        
    def __str__(self):
        return f"Normal(μ={self.mean}, σ={self.std})"


class MultivariateNormalDistribution(Distribution):
    """
    Multivariate normal distribution with given mean vector and covariance matrix.
    """
    def __init__(self, mean, cov, seed=None, dtype=float):
        super().__init__(seed, dtype)
        self.type = "numeric"
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        if self.mean.shape[0] != self.cov.shape[0] or self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError("Mean vector and covariance matrix dimensions must match")
            
    def __call__(self, n, **kwargs):
        """Generate n samples from multivariate normal distribution"""
        params = {
            'mean': kwargs.get('mean', self.mean),
            'cov': kwargs.get('cov', self.cov)
        }
        return np.random.multivariate_normal(params['mean'], params['cov'], size=n).astype(self.dtype)
        
    def __str__(self):
        return f"MultivariateNormal(μ={self.mean}, Σ=matrix{self.cov.shape})"


class BetaDistribution(Distribution):
    """
    Beta distribution with shape parameters a and b.
    """
    def __init__(self, a=1.0, b=1.0, seed=None, dtype=float):
        super().__init__(seed, dtype)
        self.type = "numeric"
        self.a = a
        self.b = b
        
    def __call__(self, n, **kwargs):
        """Generate n samples from beta distribution"""
        params = {
            'a': kwargs.get('a', self.a),
            'b': kwargs.get('b', self.b)
        }
        return np.random.beta(params['a'], params['b'], size=n).astype(self.dtype)
        
    def __str__(self):
        return f"Beta(a={self.a}, b={self.b})"


class GammaDistribution(Distribution):
    """
    Gamma distribution with shape parameter k and scale parameter theta.
    """
    def __init__(self, k=1.0, theta=1.0, seed=None, dtype=float):
        super().__init__(seed, dtype)
        self.type = "numeric"
        self.k = k
        self.theta = theta
        
    def __call__(self, n, **kwargs):
        """Generate n samples from gamma distribution"""
        params = {
            'k': kwargs.get('k', self.k),
            'theta': kwargs.get('theta', self.theta)
        }
        return np.random.gamma(params['k'], params['theta'], size=n).astype(self.dtype)
        
    def __str__(self):
        return f"Gamma(k={self.k}, θ={self.theta})"


class UniformDistribution(Distribution):
    """
    Uniform distribution between low and high values.
    """
    def __init__(self, low=0, high=1, seed=None, dtype=float):
        super().__init__(seed, dtype)
        self.type = "numeric"
        self.low = low
        self.high = high
        
    def __call__(self, n, **kwargs):
        """Generate n samples from uniform distribution"""
        params = {
            'low': kwargs.get('low', self.low),
            'high': kwargs.get('high', self.high)
        }
        return np.random.uniform(params['low'], params['high'], size=n).astype(self.dtype)
        
    def __str__(self):
        return f"Uniform(low={self.low}, high={self.high})"


class ExponentialDistribution(Distribution):
    """
    Exponential distribution with given rate parameter.
    """
    def __init__(self, rate=1.0, seed=None, dtype=float):
        super().__init__(seed, dtype)
        self.type = "numeric"
        self.rate = rate
        
    def __call__(self, n, **kwargs):
        """Generate n samples from exponential distribution"""
        params = {
            'rate': kwargs.get('rate', self.rate)
        }
        return np.random.exponential(1/params['rate'], size=n).astype(self.dtype)
        
    def __str__(self):
        return f"Exponential(rate={self.rate})"


class PoissonDistribution(Distribution):
    """
    Poisson distribution with given mean parameter lambda.
    """
    def __init__(self, lam=1.0, seed=None, dtype=float):
        super().__init__(seed, dtype)
        self.type = "numeric"
        self.lam = lam
        
    def __call__(self, n, **kwargs):
        """Generate n samples from Poisson distribution"""
        params = {
            'lam': kwargs.get('lam', self.lam)
        }
        return np.random.poisson(params['lam'], size=n).astype(self.dtype)
        
    def __str__(self):
        return f"Poisson(λ={self.lam})"


class CategoricalDistribution(Distribution):
    """
    Categorical distribution over a set of categories with given probabilities.
    """
    def __init__(self, probs, categories=None, seed=None, sequential=False):
        super().__init__(seed, dtype=str)
        self.type = "categorical"
        self.probs = np.array(probs)
        if not np.isclose(np.sum(self.probs), 1.0):
            self.probs = self.probs / np.sum(self.probs)
        
        if categories is None:
            self.categories = np.arange(len(probs))
        else:
            if len(categories) != len(probs):
                raise ValueError("Length of categories must match length of probabilities")
            self.categories = np.array(categories)

        self.sequential = sequential
        
    def __call__(self, n, probs=None, **kwargs):
        """Generate n samples from categorical distribution
        
        Args:
            n: Number of samples to generate
            probs: Optional alternative probabilities to use instead of stored ones.
                  Must match length of categories.
        """
        params = {
            'probs': kwargs.get('probs', self.probs)
        }
        probs = np.array(params['probs'])

        if len(probs) != len(self.categories):
            raise ValueError("Length of provided probabilities must match length of categories")
        if not np.isclose(np.sum(probs), 1.0):
            probs = probs / np.sum(probs)
        
        return np.random.choice(self.categories, size=n, p=probs)
        
    def __str__(self):
        return f"Categorical(probs={self.probs}, categories={self.categories})"

    def is_sequential(self):
        return self.sequential
