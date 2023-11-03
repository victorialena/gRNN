import numpy as np
import argparse, glob, queue
import acm_parser
import pdb


def generate_all_loc_vel_np(all_np_motion, seq_len): #, num_train, num_val, mode='train', trial_inds=None):
    # if trial_inds is not None:
    #     subset_np_motion = [all_np_motion[i] for i in trial_inds]
    # elif mode == 'train':
    #     subset_np_motion = all_np_motion[:num_train]
    # elif mode == 'valid':
    #     subset_np_motion = all_np_motion[num_train:num_train+num_val]
    # elif mode == 'test':
    #     subset_np_motion = all_np_motion[num_train+num_val:]
    ret_loc_all = []
    ret_vel_all = []
    for np_motion in all_np_motion: #subset_np_motion:
        ret_loc, ret_vel = generate_loc_vel_np(np_motion, seq_len)
        ret_loc_all.append(ret_loc)
        ret_vel_all.append(ret_vel)
    if seq_len != -1:
        ret_loc_all = np.concatenate(ret_loc_all)
        ret_vel_all = np.concatenate(ret_vel_all)
    return ret_loc_all, ret_vel_all


def generate_loc_vel_np(np_motion, seq_len=50):
    # pdb.set_trace()
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
    # pdb.set_trace()
    all_amc_file = sorted(glob.glob(data_path+'*.amc'))
    asf_path = glob.glob(data_path + '*.asf')[0]
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
        label_alternating_joints(joints)
        motions = acm_parser.parse_amc(amc_path)
        np_motion, joint_masks = process_amc_file(joints, motions)
        if final_joint_masks is None:
            final_joint_masks = joint_masks
        else:
            assert np.array_equal(final_joint_masks, joint_masks)
        all_np_motion.append(np_motion)
        total_frame += np_motion.shape[0]
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


def load_trial_list(trial_list_file):
    # NOTE: this assumes files are consecutively labeled from 1 to num_trials
    with open(trial_list_file, 'r') as fin:
        data = fin.readlines()
    train_trials = np.array([int(x) for x in data[0].strip().split()]) - 1
    print("TRAIN TRIALS: ",train_trials.dtype)
    val_trials = np.array([int(x) for x in data[1].strip().split()]) - 1
    test_trials = np.array([int(x) for x in data[2].strip().split()]) - 1
    return train_trials, val_trials, test_trials

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/motion/39/') #35/') #118/')
    parser.add_argument('--out_path', type=str, default='data/motion/39/') #35/') #118/')
    parser.add_argument('--num_train_trials', type=int, default=12)
    parser.add_argument('--num_val_trials', type=int, default=4)
    parser.add_argument('--train_seq_len', type=int, default=50)
    parser.add_argument('--test_seq_len', type=int, default=100)
    parser.add_argument('--trial_list_file')
    args = parser.parse_args()
    data_path = args.data_path
    out_path = args.out_path
    if args.trial_list_file is not None:
        train_trials, val_trials, test_trials = load_trial_list(args.trial_list_file)
    else:
        train_trials, val_trials, test_trials = None, None, None
    all_np_motion, joint_masks, edges = process_all_amc_file(data_path)
    train_loc, train_vel = generate_all_loc_vel_np(all_np_motion, args.train_seq_len) #, args.num_train_trials, args.num_val_trials, mode='train', trial_inds=train_trials)
    # valid_loc, valid_vel = generate_all_loc_vel_np(all_np_motion, args.train_seq_len, args.num_train_trials, args.num_val_trials, mode='valid', trial_inds=val_trials)
    # test_loc, test_vel = generate_all_loc_vel_np(all_np_motion, args.test_seq_len, args.num_train_trials, args.num_val_trials, mode='test', trial_inds=test_trials)
    
    np.save(out_path + 'all_features.npy', np.concatenate((train_loc, train_vel), axis=-1))
    
    # np.save(out_path + '%s_train_cmu.npy' % 'loc', train_loc)
    # np.save(out_path + '%s_train_cmu.npy' % 'vel', train_vel)

    # # Save valid
    # np.save(out_path + '%s_valid_cmu.npy' % 'loc', valid_loc)
    # np.save(out_path + '%s_valid_cmu.npy' % 'vel', valid_vel)

    # # Save test
    # np.save(out_path + '%s_test_cmu.npy' % 'loc', test_loc)
    # np.save(out_path + '%s_test_cmu.npy' % 'vel', test_vel)

    # Save joint masks
    np.save(out_path + 'joint_masks.npy', joint_masks)

    # Save edges
    np.save(out_path + 'edges.npy', edges)