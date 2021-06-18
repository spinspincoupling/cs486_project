function run_pose_detection(path_root, cls)

addpath('third_party/nbest_release/');

if isequal(cls, 'diving')
  pose_model = ['pose_model/Diving_final.mat'];
  min_level = 8;
elseif isequal(cls, 'figure_skating')
  pose_model = ['third_party/nbest_release/PARSE_final.mat'];
  min_level = 0;
end

path_frames = [path_root 'cached/' cls '/frames/'];

path_out = [path_root 'cached/' cls '/pose_nbest_pframe/'];

load(pose_model);
partIDs = [1 2 3 5 7 10 12 14 15 17 19 22 24 26]; % indice to joints/head

seqs = get_video_list(path_root, cls);

%%% open matlab pool
try
  matlabpool close
end
try
  matlabpool
end
  
for ii = 1:length(seqs) %%% for each video
  seq1 = seqs{ii};
  fname_annot = [path_root '/action_quality_dataset/' cls '/annotations/' seq1 '.txt'];
  [fr_s fr_e] = read_annotation(fname_annot);

  frs = [];
  for i = 1:length(fr_s);
    frs = [frs fr_s(i):fr_e(i)]; %% all action frames
  end
  
  mkdir([path_out seq1 '/']);
  
  len1 = length(frs);

  h_rand_seed(); %% randomly change random seed point
  inds = randperm(len1); %% randomly permute frames
  parfor i = 1:len1
    fr = frs(inds(i));
    fr
    run_pose_detection_once(fr, path_frames, path_out, seq1, model, partIDs, min_level, cls); %%% run detection on one frame
  end
end
