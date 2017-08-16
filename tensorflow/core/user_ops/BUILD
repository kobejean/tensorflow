package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "roll.so",
    srcs = ["roll.cc"],
    gpu_srcs = ["roll_gpu.cu.cc"],
)

py_test(
    name = "roll_test",
    size = "small",
    srcs = ["roll_test.py"],
    srcs_version = "PY2AND3",
    tags = tf_cuda_tests_tags(),
    deps = [
        ":roll.so",
        "//tensorflow:tensorflow_py",
    ],
    data = ["roll_test.py"]
)
