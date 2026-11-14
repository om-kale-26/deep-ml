import numpy as np
import math

# scipy is not available in this environment. The two helpers below compute
# the regularized incomplete beta function via the Numerical Recipes
# continued-fraction algorithm — you'll need them to compute the p-value
# from the t-distribution.
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function."""
    max_iter, eps = 200, 3e-12
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    return h

def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betai(a, b, x):
  """Regularized incomplete beta function I_x(a, b)."""
  if x < 0.0 or x > 1.0:
    raise ValueError("x must be in [0, 1]")
  if x == 0.0 or x == 1.0:
    bt = 0.0
  else:
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
  if x < (a + 1.0) / (a + b + 2.0):
    return bt * _betacf(a, b, x) / a
  else:
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def two_sample_t_test(sample1, sample2, alpha=0.05):
  s1 = np.asarray(sample1, dtype=float)
  s2 = np.asarray(sample2, dtype=float)

  n1, n2 = len(s1), len(s2)
  mean1, mean2 = np.mean(s1), np.mean(s2)

  # Sample variances (unbiased, ddof=1)
  var1 = np.var(s1, ddof=1)
  var2 = np.var(s2, ddof=1)

  # Standard error and Welch's t-statistic
  vn1 = var1 / n1
  vn2 = var2 / n2
  se = math.sqrt(vn1 + vn2)
  t_stat = (mean1 - mean2) / se

  # Welch-Satterthwaite degrees of freedom
  df = (vn1 + vn2) ** 2 / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

  # Two-tailed p-value via regularized incomplete beta function
  x = df / (df + t_stat**2)
  p_value = _betai(0.5 * df, 0.5, x)

  # Cohen's d using pooled standard deviation
  pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
  cohens_d = (mean1 - mean2) / math.sqrt(pooled_var)

  return {
      "t_statistic": round(float(t_stat), 4),
      "p_value": round(float(p_value), 6),
      "degrees_of_freedom": round(float(df), 4),
      "reject_null": bool(p_value < alpha),
      "cohens_d": round(float(cohens_d), 4),
  }