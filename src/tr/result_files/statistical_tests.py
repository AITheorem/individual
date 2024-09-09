import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportions_ztest

# s2inhibition / name -mover is more common than s2collector / passthrough
observed_heads = 100
trials = 120
h0_p = 0.5
p_value = binomtest(observed_heads, trials, h0_p, alternative="two-sided")

print(p_value)

# s2inhibition / name -mover becomes less common as we increase number of layers
counts = np.array([84, 100])
nobs = np.array([120, 120])
stat, p_value = proportions_ztest(counts, nobs, alternative="two-sided")

print(p_value)

print(84 / 120, 100 / 120)
