import os
import time
import numpy as np
from fairmotion.data import bvh

data = []
end_indices = []

start_time = time.time() # time check

dataset_path = 'dataset/bvh/ubisoft/'
dataset = ['walk1_subject1.bvh', 'run1_subject2.bvh', 'run1_subject5.bvh', 'sprint1_subject2.bvh', 'sprint1_subject4.bvh']

for file in dataset:
    data_path = dataset_path + file

    motion = bvh.load(data_path)
    end_indices.append(len(motion.poses))
    facing_direction = [motion.poses[i].get_facing_direction() for i in range(len(motion.poses))]
    ra = [facing_direction[i][0] for i in range(len(motion.poses))]

    positions = motion.positions(local=False) # (frames, joints, 3)

    velocities = positions[1:] - positions[:-1]
    orientations = motion.rotations(local=False)[..., :, :2].reshape(-1, 22, 6)

    # for i in range(positions.shape[0]):
    #     positions[:, :, 0][i] -= positions[:, 0, 0][i]
    #     positions[:, :, 1][i] -= positions[:, 0, 1][i]
    #     positions[:, :, 2][i] -= positions[:, 0, 2][i]

    # across = ((positions[:, 1] - positions[:, 5]) + (positions[:, 14] - positions[:, 18]))
    # across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    #
    # forward = np.cross(across, np.array([0, 1, 0]))
    # forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    for i in range(len(motion.poses)): # for each frames
        mini_batch = []

        x_heap = positions[i, 0, 0]
        y_heap = positions[i, 0, 1]
        z_heap = positions[i, 0, 2]

        # x_mean = np.mean(positions[i, :, 0])
        # y_mean = np.mean(positions[i, :, 1])
        # z_mean = np.mean(positions[i, :, 2])

        # positions[i, :, 0] -= x_mean
        # positions[i, :, 1] -= y_mean
        # positions[i, :, 2] -= z_mean

        positions[i, :, 0] -= x_heap
        positions[i, :, 1] -= y_heap
        positions[i, :, 2] -= z_heap

        rx = positions[:, 0, 0]
        ry = positions[:, 0, 1]
        # rz = np.arctan2(motion.rotations(local=False)[:, 0, 1, 0], motion.rotations(local=False)[:, 0, 0, 0])
        rz = positions[:, 0, 2]

        mini_batch.append(rx[i])
        mini_batch.append(ry[i])
        mini_batch.append(ra[i])

        mini_batch += positions[i].flatten().tolist()

        if i == 0:
            mini_batch += [0.0 for _ in range(66)]
        else:
            mini_batch += velocities[i - 1].flatten().tolist()

        mini_batch += orientations[i].flatten().tolist()

        data.append(mini_batch)

    now = time.time()
    elapsed = int(now - start_time)
    print("{} | frame: {} | {}m {}s".format(file, len(motion.poses), elapsed // 60, elapsed % 60))

data = np.array(data)
end_indices = np.cumsum(np.array(end_indices)) - 1

np.save('../environments/pose_ubisoft_zeroheap.npy', data[0])
np.savez('../environments/mocap_ubisoft_zeroheap', data=data, end_indices=end_indices)

end_time = time.time()
elapsed_time = int(end_time - start_time)

print('----- Total: {}m {}s -----'.format(elapsed_time // 60, elapsed_time % 60))
