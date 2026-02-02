"""
Put Option Pricer using Black-Scholes Model

This module provides functions to price European put options using the
Black-Scholes formula and calculate the Greeks.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Black-Scholes price for a European put option.

    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annual, as decimal)
    sigma : float
        Volatility of the underlying asset (annual, as decimal)

    Returns:
    --------
    float
        Put option price
    """
    if T <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return put_price


def calculate_put_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """
    Calculate the Greeks for a European put option.

    Returns:
    --------
    dict
        Dictionary containing Delta, Gamma, Vega, Theta, and Rho
    """
    if T <= 0:
        return {
            'Delta': -1.0 if S < K else 0.0,
            'Gamma': 0.0,
            'Vega': 0.0,
            'Theta': 0.0,
            'Rho': 0.0
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta: negative for puts (ranges from -1 to 0)
    delta = norm.cdf(d1) - 1

    # Gamma: same as call (rate of change of delta)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega: same as call (sensitivity to volatility)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Theta: time decay for put
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Rho: negative for puts (sensitivity to interest rate)
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }


def price_put_with_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """Calculate both the put option price and all Greeks."""
    price = black_scholes_put(S, K, T, r, sigma)
    greeks = calculate_put_greeks(S, K, T, r, sigma)

    result = {'Put Price': price}
    result.update(greeks)

    return result


def main():
    print("=" * 60)
    print("European Put Option Pricer (Black-Scholes Model)")
    print("=" * 60)

    # Example parameters
    S = 100.0  # Current stock price
    K = 105.0  # Strike price
    T = 1.0  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.20  # Volatility (20%)

    print(f"\nInput Parameters:")
    print(f"  Stock Price (S):        ${S:.2f}")
    print(f"  Strike Price (K):       ${K:.2f}")
    print(f"  Time to Maturity (T):   {T:.2f} years")
    print(f"  Risk-free Rate (r):     {r * 100:.2f}%")
    print(f"  Volatility (σ):         {sigma * 100:.2f}%")

    # Calculate price and Greeks
    results = price_put_with_greeks(S, K, T, r, sigma)

    print(f"\nResults:")
    print(f"  Put Option Price:       ${results['Put Price']:.4f}")
    print(f"\nGreeks:")
    print(f"  Delta:                  {results['Delta']:.4f}")
    print(f"  Gamma:                  {results['Gamma']:.4f}")
    print(f"  Vega:                   {results['Vega']:.4f}")
    print(f"  Theta:                  {results['Theta']:.4f} (per day)")
    print(f"  Rho:                    {results['Rho']:.4f}")

    # In-the-money example
    print("\n" + "=" * 60)
    print("In-the-Money Put Example")
    print("=" * 60)

    S_itm = 95.0
    results_itm = price_put_with_greeks(S_itm, K, T, r, sigma)

    print(f"  Stock Price:            ${S_itm:.2f}")
    print(f"  Strike Price:           ${K:.2f} (ITM)")
    print(f"  Put Option Price:       ${results_itm['Put Price']:.4f}")
    print(f"  Delta:                  {results_itm['Delta']:.4f}")


if __name__ == "__main__":
    main()