"""
Call Option Pricer using Black-Scholes Model

This module provides functions to price European call options using the
Black-Scholes formula and calculate the Greeks (Delta, Gamma, Vega, Theta, Rho).
"""

import numpy as np
from scipy.stats import norm
from typing import Dict


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Black-Scholes price for a European call option.

    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annual, as decimal, e.g., 0.05 for 5%)
    sigma : float
        Volatility of the underlying asset (annual, as decimal)

    Returns:
    --------
    float
        Call option price
    """
    if T <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """
    Calculate the Greeks for a European call option.

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
    dict
        Dictionary containing Delta, Gamma, Vega, Theta, and Rho
    """
    if T <= 0:
        return {
            'Delta': 1.0 if S > K else 0.0,
            'Gamma': 0.0,
            'Vega': 0.0,
            'Theta': 0.0,
            'Rho': 0.0
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta: rate of change of option price with respect to stock price
    delta = norm.cdf(d1)

    # Gamma: rate of change of delta with respect to stock price
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega: sensitivity to volatility (divided by 100 for 1% change)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Theta: time decay (divided by 365 for daily decay)
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365

    # Rho: sensitivity to interest rate (divided by 100 for 1% change)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }


def price_call_with_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """
    Calculate both the call option price and all Greeks.

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
    dict
        Dictionary containing the call price and all Greeks
    """
    price = black_scholes_call(S, K, T, r, sigma)
    greeks = calculate_greeks(S, K, T, r, sigma)

    result = {'Call Price': price}
    result.update(greeks)

    return result


def main():
    """
    Example usage of the call option pricer.
    """
    print("=" * 60)
    print("European Call Option Pricer (Black-Scholes Model)")
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
    results = price_call_with_greeks(S, K, T, r, sigma)

    print(f"\nResults:")
    print(f"  Call Option Price:      ${results['Call Price']:.4f}")
    print(f"\nGreeks:")
    print(f"  Delta:                  {results['Delta']:.4f}")
    print(f"  Gamma:                  {results['Gamma']:.4f}")
    print(f"  Vega:                   {results['Vega']:.4f}")
    print(f"  Theta:                  {results['Theta']:.4f} (per day)")
    print(f"  Rho:                    {results['Rho']:.4f}")

    # Additional example: At-the-money option
    print("\n" + "=" * 60)
    print("At-the-Money Example")
    print("=" * 60)

    K_atm = 100.0
    results_atm = price_call_with_greeks(S, K_atm, T, r, sigma)

    print(f"  Strike Price:           ${K_atm:.2f} (ATM)")
    print(f"  Call Option Price:      ${results_atm['Call Price']:.4f}")
    print(f"  Delta:                  {results_atm['Delta']:.4f}")


if __name__ == "__main__":
    main()
