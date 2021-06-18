function dump_frames(path_root, cls, fname_vid)
% path_root = '../';
% cls = 'diving';
% cls = 'figure_skating';

path_in = [path_root 'action_quality_dataset/' cls '/videos/'];
path_out = [path_root 'cached/' cls '/frames/'];

frame_rate = 25;
if exist('fname_vid')
  dirlist1(1).name = fname_vid;
else
  dirlist1 = dir(path_in);
  dirlist1(1:2) = [];
end

for i1 = 1:length(dirlist1)
  vid_fname = [path_in dirlist1(i1).name];
  frames_path = [path_out dirlist1(i1).name(1:end-4) '/'];
  mkdir(frames_path);
  
  cmd = ['LD_LIBRARY_PATH=/usr/lib/; ffmpeg -sameq -i ' vid_fname ' -r ' num2str(frame_rate) ' ' frames_path '%8d.jpg -sameq'];
  unix(cmd);
end
