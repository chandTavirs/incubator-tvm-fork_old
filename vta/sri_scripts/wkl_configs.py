from collections import namedtuple

Workload = namedtuple(
    "Conv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
    ],
)

DepthwiseConv2D = namedtuple(
    "DepthwiseConv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
        "groups"
    ],
)

maxPoolConfig = namedtuple(
    "MaxPool2DConfig",
    [
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
        "ceil_mode"
    ],
)

batchNormConfig = namedtuple(
    "BatchNorm2DConfig",
    [
        "num_filters"
    ]
)

reluConfig = namedtuple(
    "ReluConfig",
    [
        "num_filters"
    ]
)