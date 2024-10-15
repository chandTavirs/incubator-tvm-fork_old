from random import choice, choices
from Wkls import *
wkl_list = [(0.40, WKLS_3x1x1),
            (0.20, WKLS_3x2x0),
            (0.05, WKLS_3x2x1),
            (0.05, WKLS_3x1x0),
            (0.1, WKLS_1x2x0),
            (0.05, WKLS_1x1x0),
            (0.05, WKLS_5x2x2),
            (0.05, WKLS_5x1x2),
            (0.05, WKLS_5x1x0),
            ]

probs, wkl_lists = zip(*wkl_list)

for i in range(100):
    conv_wkl = choice(choices(wkl_lists, probs)[0])
    print(conv_wkl[1])
