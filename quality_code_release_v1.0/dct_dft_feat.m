function [feats score2_vec fr_s fr_e] = dct_dft_feat(boxes, fname_annot, len1, n_seg, feat_type)

cents_x = (boxes(:, 1:4:end-3)+boxes(:, 3:4:end-3))/2;
cents_y = (boxes(:, 2:4:end-3)+boxes(:, 4:4:end-3))/2;

[fr_s fr_e score2_vec] = read_annotation(fname_annot);

for i = 1:length(fr_s)
  frs = [fr_s(i):fr_e(i)];
  cents_x1 = cents_x(frs, :);
  cents_y1 = cents_y(frs, :);
  
  %%% normalize relative to the head location
  dist_feat = [];
  tmpx = bsxfun(@minus, cents_x1, cents_x1(:, 1));
  tmpy = bsxfun(@minus, cents_y1, cents_y1(:, 1));
  dist_feat = [dist_feat tmpx tmpy];
  
  %%% for each joint: collect absolute value in the frequency domain
  feat2 = [];
  r = length(frs)/n_seg;
  for j = 1:n_seg
    dist_feat1 = dist_feat(round((j-1)*r+1) : round(j*r), :);
    
    if isequal(feat_type, 'pose+DCT')
      feat1 = dct1(dist_feat1, len1);
    elseif isequal(feat_type, 'pose+DFT')
      feat1 = fft(dist_feat1);
      feat1 = feat1(1:len1, :);
    end
    feat2 = [feat2; abs(feat1)];
  end
  feats(:, i) = feat2(:);
end
