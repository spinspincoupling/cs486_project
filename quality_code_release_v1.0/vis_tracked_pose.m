function vis_tracked_pose(boxes_tracked1, seq1, fr_s1,  fr_e1, path_frames, cls);
colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
close all

for j = 1:size(boxes_tracked1, 1)
  fr  =fr_s1+j-1;
  fname = [path_frames seq1 '/' sprintf('%0.8d', fr) '.jpg'];
  im = imread(fname);
  
  if isequal(cls, 'diving')
    im = imresize(im, .5);
  else
    im = imresize(im, [NaN 640]);
  end
  
  showskeleton1(im, boxes_tracked1(j, :), colorset, pa)
  title(num2str(fr));
  pause(.1)
end
