import numpy as np
import torch
from fairmotion.data import bvh

raw_data = np.load('../environments/mocap_pfnn_5.npz')
data = raw_data["data"]
end_indices = raw_data["end_indices"]
print("data shape: ", data.shape)
print('------------------------------------------')
print("end_indices: ", end_indices)