import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

'''This script is meant to compare two response rates corresponding to two
different stimuli, s1 and s2. Each of these response rates can be approximated
using a Gaussain distribution, with varying parameters. We want to understand 
which of these response rates would 'make the best decision threshold' for 
determining whether a stimulus s is of the form s1 or s2. Caveat: it is twice
as dangerous to erroneously classify a stimulus as s2 than it is s1.'''

if __name__ == '__main__':
    # Gaussian parameters
    s1_mean = 5
    s1_std = 0.5
    s2_mean = 7
    s2_std = 1
    num_samples = 1000
    discount = 3

    min_mean = min(s1_mean, s2_mean)
    max_mean = max(s1_mean, s2_mean)
    dist = np.linspace(min_mean - discount, max_mean + discount, num_samples)
    s1_pdf = norm.pdf(dist, s1_mean, s1_std)
    s2_pdf = norm.pdf(dist, s2_mean, s2_std)

    # Likelihood
    # We are penalized twice as much for s2 fp's than for s1 fp's
    loss_s2 = 2
    loss_s1 = 1
    likelihood_ratio = s2_pdf/s1_pdf
    loss_ratio = loss_s2/loss_s1

    # Plot distribution (pdf vs stimuli)
    plt.figure(facecolor='white', dpi=200)
    plt.subplot(1, 2, 1)
    plt.plot(dist, s1_pdf, color='blue', label='Stimulus 1 (s1)')
    plt.plot(dist, s2_pdf, color='red', label='Stimulus 2 (s2)')
    plt.title('Stimulus conditional distributions')
    plt.xlabel('Stimulus')
    plt.ylabel('PDF')
    plt.axvline(x=s1_mean, linestyle='--', color='blue')
    plt.axvline(x=s2_mean, linestyle='--', color='red')
    plt.legend()

    # Plot likelihood ratio - what we're solving for
    plt.subplot(1, 2, 2)
    plt.plot(dist, likelihood_ratio, color='green', label='Likelihood ratio')
    plt.axhline(loss_ratio, color='green', linestyle='--')
    plt.title('Likelihood ratio')
    plt.xlabel('Stimulus')
    plt.ylabel('Likelihood ratio')
    plt.ylim((0, 4))
    plt.legend()

    # Plot
    plt.show()
    plt.close()

