function boxes_tracked = track_pose_once(boxess, max_num_detec, trans_cost);
n_frs1  =size(boxess, 3);
boxess2 = boxess(:, 1:end-3, :).^2;
boxess3 = sum(boxess2, 2);

dp_c = zeros(max_num_detec, n_frs1)-inf;
dp_c(:, 1) = -boxess(:, end, 1);
for j = 2:n_frs1
  a1 = bsxfun(@plus, boxess3(:, 1, j-1), boxess3(:, 1, j)');
  a2 = boxess(:, 1:end-3, j-1)*boxess(:, 1:end-3, j)';
  cost_tr = a1-2*a2;
  cost_tr = cost_tr/trans_cost;
  cost1 = bsxfun(@plus, cost_tr, dp_c(:, j-1));
  cost1 = bsxfun(@plus, cost1, -boxess(:, end, j)');
  [dp_c(:, j) dp_link(:, j)] = min(cost1' ,[], 2);
end
clear inds boxes_tracked
[cost_total inds(n_frs1)] = min(dp_c(:, n_frs1));
for j = n_frs1-1:-1:1
  inds(j) = dp_link(inds(j+1), j+1);
end
for j = 1:n_frs1
  boxes_tracked(j, :) = boxess(inds(j), :, j);
end

%%% for verification only  begin
cost_total2 = -boxes_tracked(1, end);
for j = 2:n_frs1
  cost_total2 = cost_total2 ...
    + sum((boxes_tracked(j-1, 1:end-3) - boxes_tracked(j, 1:end-3)).^2, 2)/trans_cost ...
    - boxes_tracked(j, end);
end
assert(max(abs(cost_total - cost_total2)) < 1e-10);
%%% for verification only  end
