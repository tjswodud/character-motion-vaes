import numpy as np
import torch

raw_data = np.load('../environments/mocap_test.npz')
# raw_data = np.load('../environments/pose0.npy')
print("data shape: ", raw_data["data"].shape)
print('------------------------------------------')
print("end_indices: ", raw_data['end_indices'])