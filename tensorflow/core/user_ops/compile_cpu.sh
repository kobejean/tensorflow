#!/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared roll.cc -o roll.so -fPIC -I $TF_INC -O2  -D_GLIBCXX_USE_CXX11_ABI=0 #-undefined dynamic_lookup
python3 roll_test.py
