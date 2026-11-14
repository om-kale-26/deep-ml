import math


def calculate_power(
    effect_size: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> float:
  """
    Calculate statistical power for a two-sample z-test.
    """

  def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

  def inv_phi(p: float) -> float:
    if p < 0.5:
      return -inv_phi(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    numerator = c0 + c1 * t + c2 * (t**2)
    denominator = 1.0 + d1 * t + d2 * (t**2) + d3 * (t**3)
    return t - numerator / denominator

  lam = effect_size * math.sqrt(sample_size_per_group / 2.0)

  if two_tailed:
    z_c = inv_phi(1.0 - alpha / 2.0)
    power = 1.0 - phi(z_c - lam) + phi(-z_c - lam)
  else:
    z_c = inv_phi(1.0 - alpha)
    power = 1.0 - phi(z_c - lam)

  return round(float(power), 4)