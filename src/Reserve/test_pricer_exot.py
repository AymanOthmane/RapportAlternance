import numpy as np
from scipy.stats import norm




















class MertonJumpDiffusionModel:
    def __init__(self, S0, K, T, r, sigma, lam, mu_j, sigma_j):
        """
        Initialize the Merton Jump-Diffusion Model.

        :param S0: Initial stock price
        :param K: Strike price
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate
        :param sigma: Volatility of the underlying asset
        :param lam: Jump intensity (average number of jumps per year)
        :param mu_j: Mean of the jump size (in log terms)
        :param sigma_j: Standard deviation of the jump size (in log terms)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def d1(self, S, K, T, r, sigma):
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def d2(self, d1, sigma, T):
        return d1 - sigma * np.sqrt(T)

    def black_scholes_call(self, S, K, T, r, sigma):
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(d1, sigma, T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def option_price(self):
        """
        Calculate the European call option price under the Merton Jump-Diffusion Model.

        :return: Option price
        """
        price = 0
        for k in range(50):  # summation index
            r_k = self.r - self.lam * (np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1) + k * (self.mu_j / self.T)
            sigma_k = np.sqrt(self.sigma ** 2 + k * (self.sigma_j ** 2 / self.T))
            poisson_prob = (np.exp(-self.lam * self.T) * (self.lam * self.T) ** k) / np.math.factorial(k)
            price += poisson_prob * self.black_scholes_call(self.S0, self.K, self.T, r_k, sigma_k)
        
        return price

# Example usage:
S0 = 100  # Initial stock price
K = 100   # Strike price
T = 1     # Time to maturity (in years)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
lam = 0.75  # Jump intensity
mu_j = -0.1  # Mean of jump size (in log terms)
sigma_j = 0.3  # Standard deviation of jump size (in log terms)

merton_model = MertonJumpDiffusionModel(S0, K, T, r, sigma, lam, mu_j, sigma_j)
option_price = merton_model.option_price()

print(f"The European call option price under the Merton Jump-Diffusion Model is: {option_price:.2f}")


import numpy as np
import pandas as pd

def identify_jumps(returns, threshold=3):
    """
    Identify jumps in the returns based on a threshold.
    
    :param returns: Array of log returns
    :param threshold: Threshold in standard deviations to define a jump
    :return: Array of jumps
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    jumps = returns[np.abs(returns - mean_return) > threshold * std_return]
    return jumps

def calibrate_jump_parameters(returns):
    """
    Calibrate jump parameters based on identified jumps in the returns.
    
    :param returns: Array of log returns
    :return: λ (jump intensity), μ_j (mean jump size), σ_j (std dev of jump size)
    """
    jumps = identify_jumps(returns)
    
    λ = len(jumps) / len(returns)  # Jump intensity
    μ_j = np.mean(jumps)  # Mean of jump size
    σ_j = np.std(jumps)  # Std deviation of jump size
    
    return λ, μ_j, σ_j

# Example using synthetic data
np.random.seed(42)
log_returns = np.random.normal(0, 0.02, 252)  # Simulating 252 trading days of log returns

# Injecting synthetic jumps into the returns
log_returns[::50] += np.random.normal(-0.1, 0.05, 5)  # Introducing 5 jumps

λ, μ_j, σ_j = calibrate_jump_parameters(log_returns)

print(f"Calibrated Jump Intensity (λ): {λ:.4f}")
print(f"Calibrated Mean Jump Size (μ_j): {μ_j:.4f}")
print(f"Calibrated Std Dev of Jump Size (σ_j): {σ_j:.4f}")
