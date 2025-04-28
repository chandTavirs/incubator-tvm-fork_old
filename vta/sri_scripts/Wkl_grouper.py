from wkl_configs import *
from Wkls import MOBILENET_V2

count_3x1x1 = 80
count_3x2x1 = 29
count_3x2x0 = 0
count_1x1x0 = 124
count_1x2x0 = 23
count_5x2x2 = 0
count_5x2x1 = 0
for i, wkl in enumerate(MOBILENET_V2):
    wkld = wkl[1]

    if wkld.hkernel == 1 and wkld.hpad == 0 and wkld.hstride == 2:
        print ('(\'workloads_{}\', Workload({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})),'.format(count_1x2x0,
                        wkld.batch, wkld.height, wkld.width, wkld.in_filter, wkld.out_filter,
                        wkld.hkernel,wkld.wkernel, wkld.hpad, wkld.wpad, wkld.hstride, wkld.wstride))
        count_1x2x0 += 1