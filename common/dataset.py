import os
import time
import numpy as np
from fairmotion.data import bvh

data = np.zeros((1, 375))
end_indices = []

start_time = time.time() # time check

# dataset_path = 'dataset/bvh/ubisoft/'
# dataset_path = 'dataset/bvh/bandai-namco-normal/'
dataset_path = 'dataset/bvh/pfnn/'
# dataset = ['walk1_subject1.bvh', 'run1_subject2.bvh', 'run1_subject5.bvh', 'sprint1_subject2.bvh', 'sprint1_subject4.bvh']
# dataset = ['dataset-2_run_active_%03d.bvh' % (i + 1) for i in range(59)]

dataset_list = os.listdir(dataset_path)

for file in dataset_list:
    data_path = dataset_path + file

    motion = bvh.load(data_path)
    end_indices.append(len(motion.poses))
    # facing_direction = [motion.poses[i].get_facing_direction() for i in range(len(motion.poses))]
    # ra = [facing_direction[i][0] for i in range(len(motion.poses))]

    positions = motion.positions(local=False) # (frames, joints, 3)

    velocities = positions[1:] - positions[:-1]
    orientations = motion.rotations(local=False)[..., :, :2].reshape(-1, 31, 6) # PFNN

    heap_vectors = positions[:, 6] # bandai namco : 18, 14
    shoulder_vectors = positions[:, 24] # bandai namco : 10, 9
    average_vectors = (heap_vectors + shoulder_vectors) / 2

    facing_vector = np.cross(average_vectors, np.array([[0, 0, 1]]))
    facing_vector = facing_vector / np.sqrt(np.power(facing_vector, 2).sum())

    root_linear_velocity = positions[1:, 0] - positions[:-1, 0]
    root_linear_velocity = np.concatenate((np.expand_dims(np.zeros((root_linear_velocity.shape[1],)), axis=0), root_linear_velocity))
    rx = root_linear_velocity[:, 0]
    ry = root_linear_velocity[:, 1]
    # rx = positions[:, 0, 0]
    # ry = positions[:, 0, 1]
    # ra = facing_vector[:, 0]
    ra = np.sqrt(np.power(rx, 2) + np.power(ry, 2)) / np.sqrt(np.power(positions[:, 0, 0], 2) + np.power(positions[:, 0, 1], 2))

    new_data = np.concatenate((rx.reshape(-1, 1), ry.reshape(-1, 1), ra.reshape(-1, 1)), axis=1)
    new_data = np.concatenate((new_data, positions.reshape(positions.shape[0], -1)), axis=1)

    velocities = np.concatenate((np.expand_dims(np.zeros((velocities.shape[1], velocities.shape[2])), axis=0), velocities)) # (frame, num of joints, 3)
    new_data = np.concatenate((new_data, velocities.reshape(velocities.shape[0], -1)), axis=1)
    new_data = np.concatenate((new_data, orientations.reshape(orientations.shape[0], -1)), axis=1)

    data = np.concatenate((data, new_data), axis=0)

    # for i in range(len(motion.poses)): # for each frames
    #     mini_batch = []
    #
    #     # x_mean = np.mean(positions[i, :, 0])
    #     # y_mean = np.mean(positions[i, :, 1])
    #     # z_mean = np.mean(positions[i, :, 2])
    #     #
    #     # positions[i, :, 0] -= x_mean
    #     # positions[i, :, 1] -= y_mean
    #     # positions[i, :, 2] -= z_mean
    #
    #     # rx = positions[:, 0, 0]
    #     # ry = positions[:, 0, 1]
    #     # # rz = np.arctan2(motion.rotations(local=False)[:, 0, 1, 0], motion.rotations(local=False)[:, 0, 0, 0])
    #     # rz = positions[:, 0, 2]
    #
    #     mini_batch.append(rx[i])
    #     mini_batch.append(ry[i])
    #     mini_batch.append(ra[i])
    #
    #     mini_batch += positions[i].flatten().tolist()
    #
    #     if i == 0:
    #         mini_batch += [0.0 for _ in range(66)]
    #     else:
    #         mini_batch += velocities[i - 1].flatten().tolist()
    #
    #     mini_batch += orientations[i].flatten().tolist()
    #
    #     data.append(mini_batch)

    now = time.time()
    elapsed = int(now - start_time)
    print("{} | frame: {} | {}m {}s".format(file, len(motion.poses), elapsed // 60, elapsed % 60))

data = np.delete(data, 0, axis=0)
# data = np.array(data)
end_indices = np.cumsum(np.array(end_indices)) - 1

np.save('../environments/pose_pfnn.npy', data[0].reshape(1, -1))
np.savez('../environments/mocap_pfnn_5', data=data, end_indices=end_indices)

end_time = time.time()
elapsed_time = int(end_time - start_time)

print('----- Total: {}m {}s -----'.format(elapsed_time // 60, elapsed_time % 60))
