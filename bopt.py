from typing import (
    Any,
    Tuple,
    Dict,
    TypedDict,
    Optional,
    Callable
)

import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
import pandas as pd



class CustomKernel:
    """
    """
    def __init__(self, cov: np.ndarray, points: np.ndarray) -> None:
        """
        Load kernel from npy file.
        """
        self.K = cov.copy() # load pre-computed covariance matrix from a npy file
        self.points = points # points from which covariance matrix was computed


    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Return the kernel k(X, Y).

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.


        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        """
        X = np.atleast_2d(X)

        if Y is None: Y = X.copy()

        cov = np.zeros((X.shape[0], Y.shape[0]))

        for i, x in enumerate(X):
            # find closest point from points to x
            x_indx = np.abs(self.points - x.item()).argmin()

            for j, y in enumerate(Y):
                # find closest point from points to y
                y_indx = np.abs(self.points - y.item()).argmin()

                cov[i, j] = self.K[x_indx, y_indx]

        return cov


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    


class MeanFunction:
    """
    """
    def __init__(self, mean: float = 0.) -> None:
        """
        """
        self.mean = mean


    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        """
        X = np.atleast_2d(X)
        return np.full_like(X, self.mean)

    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean_value={self.mean})"



class GaussianProcessRegressor:
    """
    """

    def __init__(
        self,
        mean,
        kernel,
        noise: float = 0.,
        random_state: int = 12345
    ) -> None:
        """
        """
        self.mean = mean
        self.kernel = kernel
        self.noise = noise
        self.rng = np.random.default_rng(seed=random_state)


    def posterior(
        self,
        X_test: np.ndarray,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        X_test = np.atleast_2d(X_test)
        X_train = np.atleast_2d(X_train)
        Y_train = np.atleast_2d(Y_train)

        K_tn = self.kernel(X_train, X_train) + self.noise**2 * np.eye(len(X_train))
        K_tt = self.kernel(X_test, X_train)
        K_ts = self.kernel(X_test, X_test)
        
        mu = self.mean(X_test) + (K_tt @ inv(K_tn) @ (Y_train - self.mean(X_train)))

        cov = K_ts - (K_tt @ inv(K_tn) @ K_tt.T)
        
        return mu, cov
    

    def predict(
        self,
        X_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        Y_train: Optional[np.ndarray] = None,
        return_std: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        """

        if X_train is None: # predict based on GP prior
            y_test_mean = self.mean(X_test)
            y_test_cov = self.kernel(X_test)

        else: # predict based on GP posterior
            y_test_mean, y_test_cov = self.posterior(X_test, X_train, Y_train)

            # Check if any of the variances is negative because of
            # numerical issues.
            # If yes: set the variance to 0.
            var_negative = np.diag(y_test_cov) < 0
            if np.any(var_negative):
                y_test_cov[var_negative] = 0.

        if return_std:
            y_test_std = np.sqrt(np.diag(y_test_cov)).reshape(-1, 1)
            return y_test_mean, y_test_std


        return y_test_mean, y_test_cov
    

    def sample(
        self,
        X: np.ndarray,
        n_samples: int = 1,
        **kwargs
    ) -> np.ndarray:
        """
        """
        X = np.atleast_2d(X)
        y_mean, y_cov = self.predict(X, **kwargs)
        return self.rng.multivariate_normal(y_mean.ravel(), y_cov, n_samples)
        



class LinearGridSpace:
    """
    Returns 1D grid of evenly spaced points over the specified interval.
    """
    def __init__(
        self,
        start: float = -180,
        end: float = 180,
        size: int = 200,
        random_state: Optional[int] = None
    ) -> None:
        self.points = np.linspace(start, end, size)
        self.rng = np.random.default_rng(seed=random_state)


    def sample(
        self,
        mode: str = "fixed" # "fixed" | "random"
    ) -> np.ndarray:
        """
        """
        if mode == "random": # sample a random point from a grid of points
            return self.rng.choice(self.points).reshape(-1, 1) 
        elif mode == "fixed": # return a full grid of points
            return self.points.reshape(-1, 1)
        else:
            raise ValueError(
                f'''Error: {mode} is not defined! \
                    Accepted arguments for "mode" parameter: "fixed", "random".'''
            )
        

    def __len__(self) -> int:
        return len(self.points)
    

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.points[0], self.points[-1])

        


class LogGridSpace:
    """
    Returns 1D grid of evenly spaced points on a log scale over the specified interval.
    """
    def __init__(
        self,
        start: float = -180,
        end: float = 180,
        center: float = 0,
        size: int = 50,
        base: float = 2,
        random_state: Optional[int] = None
    ) -> None:
        self.points = np.concatenate((
            -np.logspace(np.log2(abs(start)), -2, size // 2, base=base),
            np.logspace(-2, np.log2(end), size // 2, base=base)
        )) + center
        self.rng = np.random.default_rng(seed=random_state)


    def sample(
        self,
        mode: str = "fixed" # "fixed" | "random"
    ) -> np.ndarray:
        """
        """
        if mode == "random": # sample a random point from a grid of points
            return np.zeros((1, 1)) 
        elif mode == "fixed": # return a full grid of points
            return self.points.reshape(-1, 1)
        else:
            raise ValueError(
                f'''Error: {mode} is not defined! \
                    Accepted arguments for "mode" parameter: "fixed", "random".'''
            )
        
    
    def recenter(self, shift_by: float, normalize: bool = True):
        new = np.concatenate((
            -np.logspace(np.log2(abs(-180)), -2, 50 // 2, base=2),
            np.logspace(-2, np.log2(180), 50 // 2, base=2)
        )) + shift_by

        if normalize:
          self.points = np.unique(np.sort((new + 180) % 360 - 180, axis=0))

        

    def __len__(self) -> int:
        return len(self.points)
    

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.points[0], self.points[-1])



class AcquisitionFunction:
    """
    Implements Expected Improvement (EI) and Probability of Improvement (PI)
    """
    def __init__(
        self,
        kind: str = "ei", # "pi"
        eps: float = 0.
    ) -> None:
        self.eps = eps
        self.kind = kind

    
    def __call__(self, X: np.ndarray, y_best: float, surrogate, **kwargs) -> np.ndarray:
        """
        """
        mean, std = surrogate.predict(X, return_std=True, **kwargs)

        # TODO: ADD COMMENT 
        mask = std == 0.
        std[mask] = np.nan

        a = (mean - y_best - self.eps)
        z = a / std

        if self.kind == "pi":
            return norm.cdf(z)
        elif self.kind == "ei":
            return a * norm.cdf(z) + std * norm.pdf(z)
        else:
            raise ValueError(
                f'''Error: {self.kind} is not defined! \
                    Accepted arguments: "ei", "pi".'''
            )


class Sample(TypedDict):
    """
    """
    x: float
    y: float


Observation = Dict[int, Sample] 


class Observations:
    """
    """
    def __init__(self) -> None:
        self._data: Observation = {}


    def update(
        self,
        data: Sample,
        step: int,
    ):
        """
        """
        self._data[step] = data

        # TODO: Do we need to handle duplicates (if any)?


    def __len__(self):
        return len(self.data)
    

    @property
    def empty(self):
        return len(self) == 0
    

    @property
    def data(self):
        return pd.DataFrame(self._data).T
    
    
    @property
    def X(self):
        try:
            return self.data["x"].to_numpy().reshape(-1, 1)
        except KeyError:
            return None
    

    @property
    def Y(self):
        try:
            return self.data["y"].to_numpy().reshape(-1, 1)
        except KeyError:
            return None
    

    @property
    def maximum(self):
        """
        """
        if self.empty:
            return {
                "step": None,
                "X": None,
                "Y": None
            }
        

        index = np.nanargmax(self.Y)
        return {
            "step": index + 1,
            "X": self.X[index].item(),
            "Y": self.Y[index].item()
        }



class BayesianOptimizer:
    """
    """
    def __init__(
        self,
        obj_func,
        search_space,
        surrogate,
        acquisition,
    ) -> None:
        """
        """
        self.objective = obj_func
        self.space = search_space
        self.surrogate = surrogate
        self.acq = acquisition

        self.history = Observations()


    def step(
        self,
        cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        return_acq: bool = False
    ):
        """
        """
        if self.history.empty:
            x_next = self.space.sample(mode="random")
            y_next = self.objective(x_next)
            ei = None
        else:
            X_tries = self.space.sample(mode="fixed")
            current_best = self.history.maximum["Y"]

            ei = self.acq(
                X=X_tries,
                y_best=current_best,
                surrogate=self.surrogate,
                X_train=self.history.X,
                Y_train=self.history.Y
            )

            if cost_func is not None:
                ei *= cost_func(X_tries)

            max_idx = np.nanargmax(ei)

            x_next = X_tries[max_idx]
            y_next = self.objective(x_next)


        if return_acq:
            return (
                np.clip(x_next.item(), *self.space.bounds),
                y_next.item(),
                ei
            )

        
        return (
            np.clip(x_next.item(), *self.space.bounds),
            y_next.item(),
        )


    def log(
        self,
        data: Sample,
        step: int,
        verbose: bool = False
    ):
        self.history.update(data, step)
        
        # TODO: ADD MESSAGE
        if verbose: print("Message.")



class SearchStoppingCriterion:
    """
    """
    def __init__(
        self,
        kind: str,
        threshold: Optional[float] = None
    ) -> None:
        self.kind = kind
        self.threshold = threshold


    def __call__(self, optimizer: BayesianOptimizer, X: np.ndarray) -> bool:
        if self.kind == "sampling":
            samples = optimizer.surrogate.sample(
                X,
                X_train=optimizer.history.X,
                Y_train=optimizer.history.Y,
                n_samples=100
            )
            if_stop = np.sum(np.isclose(np.max(samples, axis=1), optimizer.history.maximum['Y'])) / 100 > self.threshold

        elif self.kind == "pi":
            poi = AcquisitionFunction(eps=0.0001, kind="pi")

            pi = poi(
                X=X,
                y_best=optimizer.history.maximum["Y"],
                surrogate=optimizer.surrogate,
                X_train=optimizer.history.X,
                Y_train=optimizer.history.Y
            )

            if_stop = not np.any(pi > self.threshold)

        elif self.kind == "ucb":
            mean, std = optimizer.surrogate.predict(X, optimizer.history.X, optimizer.history.Y, return_std=True)
            uncertainty = 1.96 * std # 95% confidence interval

            ucb = (mean + uncertainty).ravel()

            if_stop = np.isclose(np.max(ucb), optimizer.history.maximum["Y"])

        else:
            raise ValueError(
                f'''Error: {self.kind} is not defined! \
                    Accepted arguments: "ucb", "pi", "sampling".'''
            )

        return if_stop










