import numpy as np
import torch

raw_data = np.load('../environments/mocap_bandai_namco.npz')
data = raw_data["data"]
end_indices = raw_data["end_indices"]
# raw_data = np.load('../environments/pose0.npy')
print("data shape: ", data.shape)
print('------------------------------------------')
print("end_indices: ", end_indices)