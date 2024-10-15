from wkl_configs import *
from Wkls import ALL_TUNED_WKLS

count_3x1x1 = 0
count_3x2x1 = 0
count_3x2x0 = 0
count_1x1x0 = 0
count_1x2x0 = 0
count_5x2x2 = 0
count_5x2x1 = 0
for i, wkl in enumerate(ALL_TUNED_WKLS):
    wkld = wkl[1]

    if wkld.hkernel == 1 and wkld.hpad == 1 and wkld.hstride == 1:
        print ('(\'workloads_{}\', Workload({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})),'.format(count_3x1x1,
                        wkld.batch, wkld.height, wkld.width, wkld.in_filter, wkld.out_filter,
                        wkld.hkernel,wkld.wkernel, wkld.hpad, wkld.wpad, wkld.hstride, wkld.wstride))
        count_3x1x1 += 1