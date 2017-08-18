TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
nvcc -std=c++11 -c -o roll_gpu.cu.o roll_gpu.cu.cc \
-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o roll.so roll.cc \
roll_gpu.cu.o -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 # -lcudart
python3 roll_test.py
