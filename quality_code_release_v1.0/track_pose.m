function track_pose(path_root, cls)

path_frames = [path_root 'cached/' cls '/frames/'];
path_pose_nbest = [path_root 'cached/' cls '/pose_nbest/'];
path_out = [path_root 'tracked_pose/' cls '/'];

seqs = get_video_list(path_root, cls);

mkdir(path_out)

max_num_detec = 100;
trans_cost = 1e5;

for ii = 1:length(seqs)
  seq1 = seqs{ii};
  fname_annot = [path_root '/action_quality_dataset/' cls '/annotations/' seq1 '.txt'];
  [fr_s fr_e] = read_annotation(fname_annot);
  
  n_frs = h_count_frames([path_frames seq1 '/']);
  boxes_tracked_wholevideo = zeros(n_frs, 107);
  
  for i = 1:length(fr_s)
    i
    fr_s1 = fr_s(i);
    fr_e1 = fr_e(i);
    
    fname_boxes = [path_pose_nbest sprintf('%s_fr1_%0.8d_fr2_%0.8d.mat', seq1, fr_s1, fr_e1)];
    load(fname_boxes, 'boxess1');
    
    boxes_tracked{i} = track_pose_once(boxess1, max_num_detec, trans_cost);
    
    %%%% visualization
    %     vis_tracked_pose(boxes_tracked{i}, seq1, fr_s(i),  fr_e(i), path_frames, cls);
    
  end
  
  for i = 1:length(fr_s)
    for j = 1:size(boxes_tracked{i}, 1)
      boxes_tracked_wholevideo(fr_s(i)+j-1, :) = boxes_tracked{i}(j, :);
    end
  end
  save([path_out seq1 '.mat'], 'boxes_tracked_wholevideo');
end



