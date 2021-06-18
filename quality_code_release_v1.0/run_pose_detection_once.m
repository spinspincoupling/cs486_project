function run_pose_detection_once(fr, path_frames, path_out, seq1, model, partIDs, min_level, cls)
fname_out = [path_out seq1 '/' sprintf('%0.8d', fr) '.mat'];
if exist(fname_out) %% if it is not done yet
  return
end
save (fname_out, 'fr'); %% space holder so that other nodes don't work on it.

im = imread([path_frames seq1 '/' sprintf('%0.8d', fr) '.jpg']);
if isequal(cls, 'diving')
  im = imresize(im, .5);
elseif isequal(cls, 'figure_skating')
  im = imresize(im, [NaN 640]);
end

boxes = detect_MM_forquality(im, model, min(model.thresh,-1),partIDs, min_level);
if size(boxes, 1) < 1000 %% repeat with lower threshold
  boxes = detect_MM_forquality(im, model, min(model.thresh,-1)-.5,partIDs, min_level);
end
if size(boxes, 1) < 1000 %% repeat with even lower threshold
  boxes = detect_MM_forquality(im, model, min(model.thresh,-1)-2,partIDs, min_level);
end

[dummy ii] = sort(boxes(:,end),'descend');
boxes = boxes(ii,:);
boxes = boxes(1:min(size(boxes, 1), 1000), :); %%% keep only 1000-best results

save (fname_out, 'boxes');

% % visualize detections
% colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
% pa = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
% showboxes(im, boxes(1,:),colorset); % visualize the best-scoring detection
% showskeleton1(im, boxes(1,:), colorset, pa)

