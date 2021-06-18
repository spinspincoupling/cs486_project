function [fr_s fr_e score2_vec] = read_annotation(fname_annot)

%%% take one pass on the whole file to find actions and read frame numbers
word1 = 'A';
i = 0;
j = 0;
fid = fopen(fname_annot, 'r');
txt1 = fgetl(fid); %%%ignoring the header
txt1 = fgetl(fid);
txt1 = fgetl(fid);

while 1
  j = j+1;
  txt1 = fgetl(fid);
  if isequal(txt1, -1)
    break
  end
  
  f1 = strfind(txt1, word1);
  if isempty(f1)
    continue
  end
  try
    if isequal(txt1(f1+length(word1)), '_')
      continue
    end
  end
  i = i+1;
  fr = str2num(txt1(1:f1(1)-2));
  fr_s(i) = fr(1);
  fr_e(i) = fr(2);
  fr_ind(i) = j;
end
fclose(fid);


%%% take another pass on the whole file to read scores
word1 = 'Score';
i = 0;
j = 0;
fid = fopen(fname_annot, 'r');
txt1 = fgetl(fid); %%%ignoring the header
txt1 = fgetl(fid);
txt1 = fgetl(fid);
while 1
  j = j+1;
  txt1 = fgetl(fid);
  if isequal(txt1, -1)
    break
  end
  
  f1 = strfind(txt1, word1);
  if isempty(f1)
    continue
  end
  try
    if isequal(txt1(f1+length(word1)), '_')
      continue
    end
  end
  i = i+1;
  tmp = str2num(txt1(f1+length(word1):end));
  
  score1(i) = tmp(1);
  score1_vec(i, :) = tmp(1:end);
  
  score1_ind(i) = j;
  
end
fclose(fid);

for i = 1:length(fr_s);
  f1 = find(score1_ind > fr_ind(i), 1, 'first');
  assert(score1_ind(f1) - fr_ind(i) < 3);
  score2(i) = score1(f1);
  
  score2_vec(i, :) = score1_vec(f1, :);
  
end
score2_vec = score2_vec';
