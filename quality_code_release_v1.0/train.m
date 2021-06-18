clear
close all
path_root = '../';

addpath third_party/libsvm-3.12/matlab

rand('seed', 1000);

% cls = 'diving';
% n_seg = 1; %% number of segments. We divide long videos to multiple segments for better temporal resolution.

cls = 'figure_skating';
n_seg = 10; 

feat_type = 'pose+DCT';
% feat_type = 'pose+DFT';

C = 1e-3; %%for SVR

n_train = 100;  %% number of training examples
n_trial = 50;   %% number of trial
len1 = 50;      %% number of low frequency components

seqs = get_video_list(path_root, cls);

%%% extract features
[feats scores_vec] = load_features(cls, seqs, path_root, len1, n_seg, feat_type);

scores = scores_vec(1, :)/1;

n = length(scores); %% number of examples

%%% kfold cross validation
n1 = n_train; 
for i = 1:n_trial
  fprintf('.')
  inds = randperm(n);
  inds_train = inds(1:n1);
  inds_test = inds(n1+1:end);
  
  feats_train = feats(:, inds_train);
  scores_train = scores(inds_train);
  
  feats_test = feats(:, inds_test);
  scores_test = scores(inds_test);
  
  model = svmtrain(scores_train', feats_train', ['-s 4 -c ' num2str(C) ' -t 0 -q']);
  
  w1 = model.SVs'*model.sv_coef;      %% learned weight vector
  pred = w1'*feats_test - model.rho;  %% predicted values
  
  rho1(i) = corr(scores_test', pred', 'type', 'Spearman'); %% rank correlation
  rho_chance(i) = corr(scores_test', pred(randperm(length(pred)))', 'type', 'Spearman');
end
fprintf('\n')
average_rank_correlation = mean(rho1)

chance_average_rank_correlation = mean(rho_chance)

[scores_test_s inds1] = sort(scores_test);
pred_s = pred(inds1);
figure(10),
score3_test_s2 = (scores_test_s - mean(scores_test_s))/std(scores_test_s);
plot(score3_test_s2);
hold on
pred_s2 = (pred_s - mean(pred_s))/std(pred_s);
plot(pred_s2, 'r');
hold off

