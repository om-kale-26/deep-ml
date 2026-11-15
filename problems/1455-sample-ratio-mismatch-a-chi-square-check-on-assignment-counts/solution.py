import numpy as np
from math import exp, sqrt, pi, erfc


def chi2_sf(x, df):
    """Survival function P(X > x) of a chi-square distribution with integer df (provided)."""
    if x <= 0:
        return 1.0
    if df % 2 == 0:
        term, total = 1.0, 1.0
        for i in range(1, df // 2):
            term *= (x / 2.0) / i
            total += term
        return exp(-x / 2.0) * total
    total = erfc(sqrt(x / 2.0))
    term = sqrt(2.0 * x / pi) * exp(-x / 2.0)
    for i in range(1, (df - 1) // 2 + 1):
        if i > 1:
            term *= x / (2.0 * i - 1.0)
        total += term
    return total

def expected_counts(counts, ratios):
    total_counts = sum(counts)
    total_ratios = sum(ratios)
    return [float(total_counts * (r / total_ratios)) for r in ratios]

def srm_chi_square(counts, ratios):
    expected = expected_counts(counts, ratios)
    
    # Calculate Pearson's chi-square statistic: sum((O - E)^2 / E)
    chi2 = sum(((o - e) ** 2) / e for o, e in zip(counts, expected))
    
    # Degrees of freedom = number of arms (k) - 1
    df = len(counts) - 1
    
    # Compute p-value from the unrounded chi2 statistic
    p = chi2_sf(chi2, df)
    
    return (round(float(chi2), 4), round(float(p), 8))

def srm_check(counts, ratios, alpha=0.001):
    chi2, p = srm_chi_square(counts, ratios)
    flagged = bool(p < alpha)
    return (chi2, p, flagged)
