import sys, os

import numpy as np
import argparse, glob, queue
import acm_parser
import pdb


def generate_all_loc_vel_np(all_np_motion, seq_len): 
    ret_loc_all = []
    ret_vel_all = []
    for np_motion in all_np_motion:
        ret_loc, ret_vel = generate_loc_vel_np(np_motion, seq_len)
        ret_loc_all.append(ret_loc)
        ret_vel_all.append(ret_vel)
    if seq_len != -1:
        ret_loc_all = np.concatenate(ret_loc_all)
        ret_vel_all = np.concatenate(ret_vel_all)
    return ret_loc_all, ret_vel_all


def generate_loc_vel_np(np_motion, seq_len=50):
    loc = np_motion[1:]
    vel = np_motion[1:] - np_motion[:-1]
    if seq_len == -1:
        return loc, vel
    ret_loc = []
    ret_vel = []
    for k in range(0, np_motion.shape[0]-seq_len, seq_len):
        ret_loc.append(np.expand_dims(loc[k:k+seq_len], 0))
        ret_vel.append(np.expand_dims(vel[k:k+seq_len], 0))
    ret_loc = np.concatenate(ret_loc, 0)
    ret_vel = np.concatenate(ret_vel, 0)
    return ret_loc, ret_vel


def process_all_amc_file(data_path):
    all_amc_file = sorted(glob.glob(os.path.join(data_path, '*.amc')))
    asf_path = glob.glob(os.path.join(data_path, '*.asf'))[0]
    all_np_motion = []
    total_frame = 0
    final_joint_masks = None
    edges = []
    joint_ids = {}
    joints = acm_parser.parse_asf(asf_path)
    for joint_idx, joint_name in enumerate(joints):
        joint_ids[joint_name] = joint_idx
    for joint_idx, joint_name in enumerate(joints):
        joint = joints[joint_name]
        for child in joint.children:
            edges.append([joint_ids[joint_name], joint_ids[child.name]])

    for amc_path in all_amc_file:
        try:
            label_alternating_joints(joints)
            motions = acm_parser.parse_amc(amc_path)
            np_motion, joint_masks = process_amc_file(joints, motions)
            if final_joint_masks is None:
                final_joint_masks = joint_masks
            else:
                assert np.array_equal(final_joint_masks, joint_masks)
            all_np_motion.append(np_motion)
            total_frame += np_motion.shape[0]
        except:
            print('skipping ', amc_path, '... corrupted file')
    return all_np_motion, final_joint_masks, edges


def process_amc_file(joints, motions):
    out_array = np.zeros((len(motions), len(joints), 3))
    joint_masks = np.zeros(len(joints))
    for frame_idx in range(len(motions)):
        joints['root'].set_motion(motions[frame_idx])
        for joint_idx, joint_name in enumerate(joints):
            c0, c1, c2 = joints[joint_name].coordinate
            out_array[frame_idx, joint_idx, 0] = c0
            out_array[frame_idx, joint_idx, 1] = c1
            out_array[frame_idx, joint_idx, 2] = c2
    for joint_idx, joint_name in enumerate(joints):
        joint_masks[joint_idx] = float(joints[joint_name].is_labeled)
    return out_array, joint_masks


def label_alternating_joints(joints):
    root = joints['root']
    root.is_labeled = False
    all_children = queue.Queue()
    for child in root.children:
        all_children.put(child)
    while not all_children.empty():
        child = all_children.get()
        child.is_labeled = not child.parent.is_labeled
        for new_child in child.children:
            all_children.put(new_child)

def valid_dir(data_path):
    files = os.listdir(data_path)
    if len(files) < 3:
        return False
    return all([i[-3:] in ['amc', 'asf', 'txt'] for i in files])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/victorialena/mocap/') #35/') #118/')
    parser.add_argument('--out_path', type=str, default='/home/victorialena/mocap/dataset/') #35/') #118/')
    parser.add_argument('--seq_len', type=int, default=120)
    args = parser.parse_args()

    out_path = args.out_path
    root = args.data_dir

    try:
        os.mkdir(args.out_path)
    except:
        print('Path alrady exists... are you sure you want to continue? Press q to quit, c to continue')
        pdb.set_trace()

    features = []
    edges = []
    labels = []
    graphs = []

    for label in sorted(os.listdir(root)):
        data_path = os.path.join(root, label)

        label = label.split('_')[0]
        if os.path.isdir(data_path) and valid_dir(data_path):
            print('loading ', data_path)
            all_np_motion, joint_masks, graph = process_all_amc_file(data_path)
            train_loc, train_vel = generate_all_loc_vel_np(all_np_motion, args.seq_len)

            num_examples = train_loc.shape[0]
            features.append(np.concatenate((train_loc, train_vel), axis=-1))
            edges.extend([graph]*num_examples)
            labels.extend([label]*num_examples)
            graphs.append(graph)
            print(train_loc.shape)
    

    assert all([graphs[0] == graphs[i] for i in range(len(graphs))]), "Graphs deviate for different data sources!"
    
    np.save(out_path + 'features.npy', np.concatenate(features, axis=0))
    np.save(out_path + 'edges.npy', edges)
    np.save(out_path + 'labels.npy', labels)
