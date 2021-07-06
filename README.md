# cs486_project
For running the matlab code:

- The pose_data in the data folder contains a mini set of poses from 10 videos, these are the features that is loaded in load_features.m
- Copy the matlab code and pose_data folder to another folder and open it in matlab
- Only the train line in the run_from_scratch.m needs to be run. 
- Change the dirlist in get_video_list.m to the directory of pose data
- Change fname_feat to a path of pose data file
- Change to use score and difficulty level file (.mat file in data folder) and change the name of variable loaded ('difficulty_level' and 'overall_scores')