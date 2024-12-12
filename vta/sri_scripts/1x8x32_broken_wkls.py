from Wkls import ALL_TUNED_WKLS as pynq_wkls
from Wkls import MOBILENET_V2 as mnet_wkls


broken_pynq = ['workload_39', 'workload_44', 'workload_46', 'workload_50', 'workload_59', 'workload_60', 'workload_62', 'workload_64', 'workload_71', 'workload_72', 'workload_77', 'workload_78', 'workload_80', 'workload_92', 'workload_99', 'workload_102', 'workload_107', 'workload_111', 'workload_113', 'workload_132', 'workload_134', 'workload_138', 'workload_139', 'workload_145', 'workload_149', 'workload_150', 'workload_171', 'workload_176', 'workload_198', 'workload_205', 'workload_206', 'workload_207', 'workload_216', 'workload_223']
broken_mnet = ['workload_20', 'workload_21', 'workload_23', 'workload_27']

broken_wkls = []
count = 0
for _, wkl in enumerate(pynq_wkls):
    if wkl[0] in broken_pynq:
        broken_wkls.append((f'workload_{count}', wkl[1]))
        count += 1

for _, wkl in enumerate(mnet_wkls):
    if wkl[0] in broken_mnet:
        broken_wkls.append((f'workload_{count}', wkl[1]))
        count += 1

# print the broken wkls in this format - (workload_0, Workload(1, 149, 149, 32, 32, 3, 3, 0, 0, 1, 1))
for wkl in broken_wkls:
    str_prnt = f'(\'{wkl[0]}\', Workload({wkl[1][0]}, {wkl[1][1]}, {wkl[1][2]}, {wkl[1][3]}, {wkl[1][4]}, {wkl[1][5]}, {wkl[1][6]}, {wkl[1][7]}, {wkl[1][8]}, {wkl[1][9]}, {wkl[1][10]})),'
    print(str_prnt)

# create array of broken wkls[1]
broken = [wkl[1] for wkl in broken_wkls]
print(broken[0] in broken)
