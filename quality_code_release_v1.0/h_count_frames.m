function n_frs = h_count_frames(path1, pat)
if ~exist('pat')
  pat = '%0.8d.jpg';
end

l = 1;
u = 1e7;
while u > l+1
  i = round(mean((u+l)/2));
  if exist([path1 sprintf(pat, i)])
    l = i;
  else
    u = i;
  end
end
if exist([path1 sprintf(pat, u)])
  n_frs = u;
else
  n_frs = l;
end

assert(exist([path1 sprintf(pat, n_frs)]) > 0)
assert(exist([path1 sprintf(pat, n_frs+1)]) ==0)





