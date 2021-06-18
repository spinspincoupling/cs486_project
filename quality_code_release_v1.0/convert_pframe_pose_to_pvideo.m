function convert_pframe_pose_to_pvideo(path_root, cls)

% path_root = '../';
% cls = 'diving';
% cls = 'figure_skating';

N = 100; %%keep only 100-best

path_frames = [path_root 'cached/' cls '/frames/'];
path_pose_nbest_pframe = [path_root 'cached/' cls '/pose_nbest_pframe/'];
path_out = [path_root 'cached/' cls '/pose_nbest/'];

mkdir(path_out);

seqs = get_video_list(path_root, cls);

h_rand_seed(); %% not necessary if you don't run this in parallel

for ii = 1:length(seqs)
  seq1 = seqs{ii};
  fname_annot = [path_root '/action_quality_dataset/' cls '/annotations/' seq1 '.txt'];
  [fr_s fr_e] = read_annotation(fname_annot);
  
  inds = randperm(length(fr_s));
  for i = inds %% for each video
    [ii i]
    fr_s1 = fr_s(i);
    fr_e1 = fr_e(i);
    
    fname_out = [path_out sprintf('%s_fr1_%0.8d_fr2_%0.8d.mat', seq1, fr_s1, fr_e1)];
    if exist(fname_out) %% ignore if already done
      continue
    end
    save (fname_out, 'i');
    
    n_frs1 = fr_e1 - fr_s1+1;
    boxess1 = zeros(N, 107, n_frs1);
    jj = 0;
    for j = fr_s1:fr_e1
      jj = jj+1;
      fname = [path_pose_nbest_pframe seq1 '/' sprintf('%0.8d', j) '.mat'];
      tmp = load(fname, 'boxes');
      boxess1(:, :, jj) = tmp.boxes(1:N, :);
      if mod(jj, 100) == 0
        jj
      end
    end
    save(fname_out, 'boxess1', 'fr_s1', 'fr_e1');
  end
end
