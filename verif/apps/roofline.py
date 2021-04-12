#!/usr/bin/env python3

# Modified by contributors from Intel Labs

"""
Roofline Performance Model.
"""

from glob import glob
from json import load
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from verif import home

class RoofLine():

  def __init__(self, args):
    self.args = args
    self.match = args.match
    self.perf, self.bw = args.perf, args.bw
    self.ops_per_alu = args.ops_per_alu
    self.ops_per_gemm = args.ops_per_gemm
    self.font_size = args.font_size
    self.marker_size = args.marker_size
    self.tiles = None
    self.lines = None
    self.load_tiles()
    if self.tiles is not None:
      self.install_roof(self.perf, self.bw)
      if not args.no_plot:
        self.plot()

  def __repr__(self):
    return (f'<roofline:{self.match}, perf:{self.perf}, '
            f'bw:{self.bw}>')

  def load_tiles(self):
    tiles = {}
    if self.match.find('*') == -1:
      match = f'*{self.match}*' if self.match else '*'
    else:
      match = self.match
    match = match.replace('**', '*')
    profs = glob(f'{home()}/work/{match}-{self.args.target}].prof')
    no_ops = set()
    for fn in profs:
      with open(fn) as fp:
        record = load(fp)
        stem = f"{record['test']}:{record['knob']}"
        if stem not in tiles:
          tiles[stem] = dict(bytes=0, ops=0, cycles=0)
        tile = tiles[stem]
        for k, v in record.items():
          if k.endswith('nbytes'):
            tile['bytes'] += v
          elif k == 'alu_counter':
            tile['ops'] += v * self.ops_per_alu
          elif k == 'gemm_counter':
            tile['ops'] += v * self.ops_per_gemm
          elif k.startswith('cycle_count'):
            tile['cycles'] += v
    if tiles:
      for k, v in tiles.items():
        if v['ops'] == 0:
          no_ops.add(k)
      for tile in no_ops:
        del tiles[tile]
        print('Removed zero ops workload', tile)
      tiles = pd.DataFrame(tiles.values(), index=tiles.keys())
      tiles = tiles.assign(oi=tiles['ops']/tiles['bytes'])
      tiles = tiles.assign(ap=tiles['ops']/tiles['cycles'])
      tiles = tiles.assign(bw=tiles['bytes']/tiles['cycles'])
      self.tiles = tiles
    
  def install_roof(self, perf, bw):
    min_oi = min(1, min(self.tiles['oi']))
    max_oi = max(2**8, max(self.tiles['oi']))
    self.lines = pd.DataFrame(
      ((min_oi, min_oi*bw), (perf/bw, perf), (max_oi, perf)),
      index=('bottom', 'ridge', 'top'),
      columns=('oi', 'ap')
    )

  def plot(self):
    plt.rcParams.update({'font.size': self.font_size,
                        'font.weight': 'bold',
                        'axes.labelweight': 'bold',
                        'axes.titleweight': 'bold'})
    _, ax = plt.subplots(figsize=(12,6))
    ax.set_title('VTA Tensor Accelerator')
    ax.set_xlabel('Operational Intensity [ops/byte]')
    ax.set_ylabel('Available Performance [ops/cycle]')
    ax.plot(self.lines['oi'], self.lines['ap'], 'b-')
    for k, r in self.tiles[['oi', 'ap']].iterrows():
      ax.plot(r[0], r[1], 'o', markersize=self.marker_size, label=k.split(':')[1])
    #plt.plot(self.tiles['oi'], self.tiles['ap'], 'ro')
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=2)
    oi = self.tiles['oi']
    lmin = min(0, int(np.floor(np.log2(np.min(oi)))))
    lmax = max(8, int(np.ceil(np.log2(np.max(oi)))))
    ax.set_xticks([2**i for i in range(lmin, lmax+1)])
    ax.legend()
    ax.grid()
    plt.show()

if __name__ == '__main__':
  import sys
  from argparse import ArgumentParser
  if sys.argv[0].endswith('roofline.py'):
    ap = ArgumentParser(description='RoofLine Performance Model.')
    ap.add_argument('-m', '--match', type=str, default='resnet',
      help='Workload match string (default: %(default)s).')
    ap.add_argument('-p', '--perf', type=float, default=256,
      help='Top performance [ops/cycle] (default: %(default)d).')
    ap.add_argument('-b', '--bw', type=float, default=6,
      help='Memory bandwidth [bytes/cycle] (default: %(default)d).')
    ap.add_argument('-a', '--ops-per-alu', type=float, default=16,
      help='Operations per ALU loop [ops] (default: %(default)d).')
    ap.add_argument('-g', '--ops-per-gemm', type=float, default=256,
      help='Operations per GEMM loop [ops] (default: %(default)d).')
    ap.add_argument('-fs', '--font-size', type=float, default=14,
      help='Font size of labels in roofline plot (default: %(default)d).')
    ap.add_argument('-ms', '--marker-size', type=float, default=8,
      help='Marker size of points in roofline plot (default: %(default)d).')
    ap.add_argument('-n', '--no-plot', action='store_true',
      help='Do not plot (default: %(default)d).')
    ap.add_argument('-l', '--list-data', action='store_true',
      help='Print roofline data (default: %(default)d).')
    ap.add_argument('-t', '--target', type=str, default='tsim',
      choices=('fsim', 'tsim', 'bsim', 'de10nano', 'pynq'),
      help='Target to use in analysis (default: %(default)s).')
    args = ap.parse_args()
    rl = RoofLine(args)
    if (args.list_data):
      pd.options.display.float_format = '{:,.2f}'.format
      print(rl.tiles)
