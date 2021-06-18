%%% this code visualizes the tracked pose for a sample diving instance.
%%% frames for all other instanances will be extracted by running "run_from_scratch.m"

clear
colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
load('../tracked_pose/diving/Diving_-_Men_10_Prel._-_London_2012_Olympic_Game__eEKo5bGe5bU.mat');
for i = 11225:11404  %% these are frames corresponding to one instance of diving.
  im = imread(['../cached/diving/frames/Diving_-_Men_10_Prel._-_London_2012_Olympic_Game__eEKo5bGe5bU/' sprintf('%0.8d.jpg', i)]);
  showskeleton1(im, boxes_tracked_wholevideo(i, :)*2, colorset, pa);  %% note that we multiply pose coordinates by 2.
  title('press any key to continue');
  pause
end
