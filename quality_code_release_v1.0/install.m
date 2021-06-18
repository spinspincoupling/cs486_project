%%% installation
mex -O h_getpid.c
cd1 = cd;
cd('third_party/nbest_release/')
compile
cd('third_party/libsvm-3.12/matlab/')
make
cd(cd1)
