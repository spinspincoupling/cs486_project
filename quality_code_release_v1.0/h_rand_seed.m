function h_rand_seed()
rand('seed', mod(double(h_getpid)+sum(h_hostname+1-1)+round(sum(clock)), 1e6));