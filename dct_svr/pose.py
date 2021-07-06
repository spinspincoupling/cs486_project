import scipy.io as sio
import numpy as np
import os


def load_pose(dir):
    # num_videos = len([name for name in os.listdir(dir) if name.endswith(".mat")])
    clips = []
    for filename in os.listdir(dir):
        if filename.endswith(".mat"):
            clip = sio.loadmat(os.path.join(dir, filename))['boxes_tracked']
            clips.append(clip)
    return clips


if __name__ == '__main__':
    pose_dir = '../data/pose_data'
    # Each clip in clips contains boxes that is passed to dct_dft_feat
    clips = load_pose(pose_dir)
