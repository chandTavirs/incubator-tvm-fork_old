from collections import namedtuple

Conv2DWorkload = namedtuple(
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

MaxPool2DConfig = namedtuple(
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

BatchNorm2DConfig = namedtuple(
    "BatchNorm2DConfig",
    [
        "num_filters"
    ]
)

ReluConfig = namedtuple(
    "ReluConfig",
    [
        "num_filters"
    ]
)