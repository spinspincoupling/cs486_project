function seqs = get_video_list(path_root, cls)
dirlist = dir([path_root 'action_quality_dataset/' cls '/videos/']);
dirlist(1:2) = [];
for i = 1:length(dirlist)
  seqs{i} = dirlist(i).name(1:end-4);
end
