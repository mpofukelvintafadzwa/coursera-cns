"""
Quiz 2 code.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
from compute_sta import compute_sta

FILENAME = 'c1p8.pickle'

with open(FILENAME, 'rb') as f:
    data = pickle.load(f)
    
stim = data['stim']
rho = data['rho']

sampling_rate = 500
sampling_period = int(round(1000/500)) # in ms

window_width = 300
num_timesteps = int(window_width/sampling_period)  # 150

spike_indices=[]
for i in range(len(rho)):
    if rho[i]==1:
        spike_indices.append(i)
num_total_spikes = len(spike_indices)  # 53601

# But only spikes recorded after first 300 ms are relevant
# Time between samples = sampling_period = 2 ms
# Ergo ignore first 150 data points (num timesteps = 150)
relevant_rho = rho[num_timesteps:]
assert len(relevant_rho) + num_timesteps == len(rho), "Did slicing wrong."

num_relevant_spikes = np.count_nonzero(rho[num_timesteps:])  # 53583

# Double checking
spikes_in_beginning = 0
first_300_rho = rho[:151]
for i in range(len(first_300_rho)):
    if first_300_rho[i]==1:
        spikes_in_beginning += 1  # 18

assert spikes_in_beginning + num_relevant_spikes == num_total_spikes

sta = compute_sta(stim, rho, num_timesteps)
time = (np.arange(-num_timesteps, 0) + 1) * sampling_period

plt.subplot(2, 1, 1)
plt.plot(time, sta)
plt.xlabel('Time (ms)')
plt.ylabel('Stimulus')
plt.title('Spike-triggered average: constant positive stimuli produce spikes.')

plt.show()
