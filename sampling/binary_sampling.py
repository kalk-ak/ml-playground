from typing import Callable, List
import numpy as np
from numpy.typing import NDArray


def binary_sampling(
    CDF: Callable[[float], float], start: float, end: float, N: int
) -> NDArray[np.float64]:
    """Generates random samples from a distribution using numerical inversion.

    This function for a given cumulative distribution function (CDF), it generates a uniform random
    probability `u` and then uses a binary search algorithm to find the value
    `x` such that CDF(x) is approximately equal to `u`. This process is
    repeated N times to generate the desired number of samples.

    Args:
        CDF (Callable[[float], float]): The Cumulative Distribution Function of the
            desired probability distribution. It must be a monotonic function
            that takes a single float `x` and returns its cumulative probability.
        start (float): The lower bound of the search interval for the random
            variable's value.
        end (float): The upper bound of the search interval.
        N (int): The total number of random samples to generate.

    Returns:
        NDArray[np.float64]: A Numpy array of shape (N,) containing the
        generated samples from the distribution.

    Example:
        >>> import scipy.stats as stats
        >>> # Generate 5 samples from a standard normal distribution.
        >>> # We'll search for values between -4 and 4.
        >>> samples = binary_sampling(stats.norm.cdf, start=-4.0, end=4.0, N=5)
        >>> print(samples.shape)
        (5,)
    """
    # Generate N random probabilities to serve as targets for the CDF
    uniform_dist: NDArray[np.float64] = np.random.uniform(low=0, high=1, size=N)
    sample: List[float] = []

    assert end >= start, (
        "The 'end' of the search range cannot be less than the 'start'."
    )

    # For each target probability, use a binary search to find the corresponding x value
    for probability in uniform_dist:
        low: float = start
        high: float = end

        # Continue narrowing the search interval until it is smaller than the tolerance
        while (high - low) > 1e-4:
            mid_point: float = (low + high) / 2

            # Compare the CDF at the midpoint with the target probability
            # If the CDF is too low, our guess for x is too low, so we search the upper half.
            if CDF(mid_point) < probability:
                low = mid_point
            # Otherwise, our guess for x is too high, so we search the lower half.
            else:
                high = mid_point

        # The final sample is the midpoint of the converged interval
        sample.append((high + low) / 2)

    return np.array(sample)
