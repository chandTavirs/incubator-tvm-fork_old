# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Created my contributors from Intel Labs

import json
import subprocess
from itertools import product


def roundup(n, k):
    return (n + k - 1)//k

def split_in_k( k, n):
    largest = roundup(n, k)
    kk = roundup(n, largest)

    extra = kk * largest - n

    if extra == 0:
        return tuple([(largest, kk)])
    else:
        return tuple([(largest, kk-extra), (largest-1, extra)])


def test_A():
    assert split_in_k( 2, 112) == ((56,2),)
    assert split_in_k( 3, 112) == ((38,1), (37,2))
    assert split_in_k( 4, 112) == ((28,4),)
    assert split_in_k( 5, 112) == ((23,2), (22,3))
    assert split_in_k( 6, 112) == ((19,4), (18,2))
    assert split_in_k( 7, 112) == ((16,7),)

    assert split_in_k(14, 112) == ((8,14),)
    assert split_in_k(15, 112) == ((8,14),)

    assert split_in_k(38, 112) == ((3,36),(2,2))
    assert split_in_k(39, 112) == ((3,36),(2,2))

def getlen( lst):
    return sum( k for _, k in lst)

def main():
    already_done = {}

    p = []

    if False:
        config = '32x32'
        #config = '16x16'
        layer = 'D1'
        n = 112
        stride = 1
    else:
        config = '32x32'
        #config = '16x16'
        layer = 'D2'
        n = 112
        stride = 2

    lb = 1
    ub = n//stride

    for th, tw in product(range(lb,ub+1),range(lb,ub+1)):

        th_lst = split_in_k(th, n//stride)
        tw_lst = split_in_k(tw, n//stride)

        key = (th_lst, tw_lst)

        if key in already_done:
            j = already_done[key]
        else:

            oh = roundup(n//stride, getlen(th_lst))
            ow = roundup(n//stride, getlen(tw_lst))

            print( f'th: {th} tw: {tw} th_lst: {th_lst} tw_lst: {tw_lst}')

            fn = f"stats-{oh}_{ow}.json"
            result = subprocess.run(["python", "depthwise/vta_depthwise_tb_tiled.py", "-w", f"mobilenet.{layer}", "-b", "2", "--oh_tile", str(oh), "--ow_tile", str(ow), "-o", fn])

            if result.returncode == 0:
                with open(fn, "rt") as fp:
                    j = json.load(fp)
            else:
                j = None

            already_done[key] = j

        if j is not None:

            j = j.copy()

            assert len(j['oh_lst']) == getlen(th_lst), f'oh: {oh} th: {th}'
            assert len(j['ow_lst']) == getlen(tw_lst), f'ow: {ow} tw: {tw}'

            j['th_tile'] = th
            j['tw_tile'] = tw

            p.append( j)

    with open(f"stats_dense-{layer}-{config}-balanced-fixed-bins2.json", "wt") as fp:
        json.dump(p, fp=fp, indent=2)

if __name__ == "__main__":
    main()
