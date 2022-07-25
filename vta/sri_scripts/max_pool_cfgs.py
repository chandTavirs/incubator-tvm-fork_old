from collections import namedtuple

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

MAX_POOL_CONFIGS = [
    ("Config_1", maxPoolConfig(3, 3, 0, 0, 2, 2, 1)),
    ("Config_2", maxPoolConfig(3, 3, 1, 1, 1, 1, 1)),
    ("Config_3", maxPoolConfig(3, 3, 1, 1, 2, 2, 0)),
    ("Config_4", maxPoolConfig(2, 2, 0, 0, 2, 2, 1))

]