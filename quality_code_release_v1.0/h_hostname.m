function host1 = h_hostname
[dummy host1] = unix('hostname');
host1 = host1(1:end-1);
