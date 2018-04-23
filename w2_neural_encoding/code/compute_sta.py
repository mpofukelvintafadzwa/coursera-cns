"""
Created on Wed Apr 22 15:21:11 2015

@author: rkp

Code to compute spike-triggered average.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def compute_sta(stim, rho, num_timesteps):
    """Compute the spike-triggered average from a stimulus and spike-train.
    Args:
        stim: stimulus time-series
        rho: spike-train time-series
        num_timesteps: how many timesteps to use in STA
    Returns:
        spike-triggered average for specified number of timesteps before spike"""
    
    sta = np.zeros((num_timesteps,))

    # This command finds the indices of all of the spikes that occur after 300 ms into the recording.
    spike_times = rho[num_timesteps:].nonzero()[0] + num_timesteps

    # Fill in this value. Note that you should not count spikes that occur before 300 ms into the recording.
    num_spikes = np.count_nonzero(rho[num_timesteps:])  # 53583

    # Compute the spike-triggered average of the spikes found. To do this, compute the average of all of the vectors
    # starting 300 ms (exclusive) before a spike and ending at the time of the event (inclusive). Each of these vectors
    # defines a list of samples that is contained within a window of 300 ms before each spike. The average of these
    # vectors should be completed in an element-wise manner.
    # i call this 'buffer time'
    buffer_time = 300 # ms
    sampling_period = 2
    # Therefore there are 300/2 = num timesteps seconds to ignore!
    for spike_time in spike_times:
        sta += stim[spike_time - num_timesteps + 1:spike_time + 1]

    assert len(spike_times) == num_spikes  # Making sure im not crazy.
    sta /= num_spikes

    return sta


