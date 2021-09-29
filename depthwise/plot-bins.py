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

# Created by contributors from Intel Labs

import numpy as np
import pandas as pd

import math
import json
import plotly.express as px
import plotly.graph_objects as go
from itertools import product

config = '32x32'
scratchpad = 512
#config = '16x16'
#scratchpad = 1024

if True:
    layer = 'D1'

    n = 112
    stride = 1
else:
    layer = 'D2'
    n = 112
    stride = 2

df_dense = pd.read_json(f'stats_dense-{layer}-{config}-balanced-fixed-bins.json')
#df_dense = pd.read_json('stats_dense.json')

#df = df.groupby( by='oh_tile').min()
#df['oh_tile'] = df.index

#fig = px.scatter_3d(df_dense, x='oh_tile', y='ow_tile', z='cycle_counter')
#fig.show()

df = df_dense

ps = np.linspace(1,n//stride,n//stride)
X,Y = np.meshgrid(ps, ps)

df['ow'] = [lst[0] for lst in df['ow_lst']]
df['oh'] = [lst[0] for lst in df['oh_lst']]

def roundup(n, k):
    return (n + k - 1)//k

def find_closest(n, x):
    return roundup(n, roundup(n, x))

def area_constraint(x):
    if stride == 1:
        # 1024 >= (x + 2) * (y + 2)
        y = scratchpad/(x+2) - 2
        return None if (x+2) > 58 or y > n else y
    else:
        # 1024 >= (2x + 1) * (2y + 1)
        y = (scratchpad/(2*x+1) - 1)/2
        return None if (2*x+1) > 58 or (2*y+1) > n else y
        


Z = []
for a, b in zip(X,Y):
    row = []
    for x,y in zip(a,b):

#        df0 = df[ np.logical_and(find_closest(n//stride,x) == df['ow'], find_closest(n//stride,y) == df['oh'])]
        df0 = df[ np.logical_and(x == df['ow'], y == df['oh'])]
        if not df0.empty:
            z = df0['cycle_counter'].iloc[0]
            row.append(z)
        else:
            row.append(None)
    Z.append(row)

def g(v):
    return (1,0) if v is None else (0,v)

minZ = min( (min(row, key=g) for row in Z), key=g)
print(f'minZ: {minZ}')

def f(v):
    if v is None:
        return v
    else:
        vv = math.pow((v - minZ)/minZ, 1)
        return vv

newZ = [ [ f(v) for v in row] for row in Z]

constraint_y = [area_constraint(x) for x in ps]


fig = go.Figure(data=[go.Contour(x=ps,y=ps,z=newZ,colorscale='tealgrn',contours=dict(start=0,end=.1),contours_coloring='heatmap'),
                      go.Scatter(x=ps,y=constraint_y,mode='lines',showlegend=False)])
fig.update_yaxes(scaleanchor='x', scaleratio=1)
fig.update_layout(width=800,height=800)

fig.show()
