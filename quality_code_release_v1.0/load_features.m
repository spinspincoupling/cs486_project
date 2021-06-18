function [feats scores_vec fr_s fr_e boxes_tracked_wholevideo] = load_features(cls, seqs, path_root, len1, n_seg, feat_type)

feats = [];
scores_vec = [];

for i = 1:length(seqs)
  fname_feat = [path_root 'tracked_pose/' cls '/' seqs{i} '.mat'];
  fname_annot = [path_root '/action_quality_dataset/' cls '/annotations/' seqs{i} '.txt'];
  load(fname_feat, 'boxes_tracked_wholevideo');
  [feats1 score_vec1 fr_s fr_e] = dct_dft_feat(boxes_tracked_wholevideo, fname_annot, len1, n_seg, feat_type);
  feats = [feats feats1];
  scores_vec = [scores_vec score_vec1];
end

