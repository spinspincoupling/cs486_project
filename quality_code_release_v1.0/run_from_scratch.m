clear

install;

path_root = '../';

cls = 'diving';
% cls = 'figure_skating';

%%%%%%%%
dump_frames(path_root, cls); %% convert videos to frames

%%% If necessary, please modify the ffmpeg command for your setting.

%%%%%%%%
%%% this can run in parallel on many different cluster nodes. 
%%% Simply run this line on any node that has access to the data. 
%%% It takes almost 10 seconds per frame per cpu.
run_pose_detection(path_root, cls); %% detect 100-best human pose on any frame

%%% should wait for all jobs to finish before continueing
%%% failed or ongoing jobs will produce a small file (<1kb). Delete them if any and run
%%% the above command again to finish them all.

%%%%%%%%
%%% this can also run in parallel on multiple nodes. 
%%% It is not that slow though.
convert_pframe_pose_to_pvideo(path_root, cls); %% aggregate results detected poses into one file for each action instant

%%%%%%%%
track_pose(path_root, cls); %% track with dynamic programming to smooth pose through time. The result contains only one pose per frame.

%%%%%%%%
train %% train and test SVR

