This is the implementation for the following paper:
Hamed Pirsiavash, Carl Vondrick, Antonio Torralba, "Assessing the Quality of Actions", ECCV 2014.

Please cite this paper if you use this code.

%%%%%%%%%%
Installation:
- Download "quality_code_release_v1.0.tar" and extract it into a directory.
- Compile the mex files by running "install" in MATLAB.

%%%%%%%%%%
Usage:
Please download the following two files and untar them into the same directory as above:

"action_quality_dataset.tar" contains 
- videos for "diving" and "figure skating". Almost 150 instances for each.
- annotations (starting and ending frame number for each action instance)

"tracked_pose.tar" contains 
- tracked human pose in all videos. This will be used in extracting features in "train.m". Alternatively, this can be reproduced using the provided code.

Run "train.m" to see the results. It reads tracked poses, then trains and tests the SVR. It uses the pre-computed, tracked human pose.

%%%%%%%%%%
Other files:
- "run_from_scratch.m": reproduces the results from videos and annotations. 
It loads videos and annotations, detects human pose, and tracks it to reproduce the data provided in "tracked_pose.tar". It runs "train.m" in the end Run this code only if you want to reproduce human pose results. This takes a long time so we recommend running it in parallel on multiple cores. This file contains description for some other mfiles.

- "visualize_pose.m" visualizes the tracked pose for an example instance of diving. Note that pose coordinates should be multiplied by two to match the frame size. 
To run it, you need to download frames for that instance from here:
http://people.csail.mit.edu/hpirsiav/codes/quality/cached_example.tar
To save space, we don't provide frames for all instances. 
"run_from_scratch.m" uses "ffmpeg" to extract the frames from all videos.

- third_party/libsvm-3.12/: libsvm code downloaded from http://www.csie.ntu.edu.tw/~cjlin/libsvm/

- third_party/nbest_release/: n-best inference on human pose estimation. It is downloaded from http://www.ics.uci.edu/~iypark/code/nbest_release.zip. 
Associated paper: Dennis Park, Deva Ramanan, "N-best maximal decoders for part models", ICCV'11.

- code/pose_model/Diving_final.mat: human pose model tuned for "diving" action. This will be used in reproducing pose estimation. For "figure skating" we use the model trained on PARSE dataset that comes with the n-best code. 

%%%%%%%%%%
Some notes:
- The frames are extracted at 25 fps (see "dump_frames.m")
- For diving they are resized to half size 
- For figure skating, they are resize to have 640 pixels in X axis. (see "run_pose_detection_once.m")
- Each pose is represented with 107 numbers: 
[104 for the location of 26 joints, 
 one for the mixture number, 
 one for the pyramid level, 
 and one for the score] 
(see "showskeleton1.m")  

