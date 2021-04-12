"""
Trace manager module with class and function implementations.
Note that instantiation requires a prior loading or the VTA
runtime shared library. Note that after the library has loaded
it will retain whatever value has been set in the manager, even
if a new manager instance is created since its static nature.
"""

# Modified by contributors from Intel Labs

import os, builtins, ctypes, json
from time import time
from contextlib import contextmanager
from tvm import get_global_func as gfunc
from .test_utils import home, sample_targets, trace_path, is_remote

# These are set by pytest and used to share command line selection.
selected_mode = 'quiet'
selected_targets = []
random_seed = 0xCafeFace

# These are set by us here.
supported_targets = {'cpu', 'bsim', 'fsim', 'tsim', 'de10nano', 'pynq'}
remote_targets = {'de10nano', 'pynq'}
unsupported_targets = {'ultra96'}
redirected_targets = {'bsim'}

def trace_targets():
  return sample_targets().union(supported_targets).difference(unsupported_targets)

t2d = {k:'vta' for k in trace_targets()}
t2d['cpu'] = 'arm_cpu'
t2t = {k:k for k in trace_targets()}
t2t['cpu'] = 'fsim'

def node_to_target(nodeid: str):
  if nodeid is not None:
    s = nodeid.replace('[', '-')[:-1]
    return s[s.rfind('-')+1:]
  return ""

class TraceMgr(ctypes.Structure):
  '''
  Access class of corresponding class in vta runtime namespace.
  Fields need to be updated whenever the class footprint changes.
  '''
  _fields_ = [
    ('fd', ctypes.c_int),
    ('trace', ctypes.c_bool),
    ('host', ctypes.c_bool),
    ('alloc', ctypes.c_bool),
    ('load', ctypes.c_bool),
    ('store', ctypes.c_bool),
    ('insn', ctypes.c_bool),
    ('prof', ctypes.c_bool),
    ('xy', ctypes.c_bool),
    ('virt', ctypes.c_bool)
  ]

  def __new__(cls, node=None, mode=None):
    '''Associate instance to C++ class object pointer.'''
    if not is_remote() and node_to_target(node) in remote_targets:
      return super(TraceMgr, cls).__new__(cls)
    vp = gfunc('vta.runtime.trace.init')(-1)
    return TraceMgr.from_address(vp.value)

  def __init__(self, node=None, mode=None):
    '''
    Initialize trace instance.
    Collect registered member functions here.
    At this point members from C++ are already present
    including self.fd, not explicitly set in __new__.
    '''
    self.target = node_to_target(node) if node else None
    self.is_client = not is_remote() and self.target in remote_targets
    self.node = node
    self.mode = mode
    self.m_enable_event = set()
    self._decode_mode()
    self._collect_functions()
    if node and mode and self.trace and not self.is_client:
      fp = trace_path(node, mode)
      flags = os.O_WRONLY|os.O_CREAT|os.O_TRUNC
      fd = os.open(fp, flags, 0o640)
      self.trace_init(fd)
      if self.target in redirected_targets:
        self.trace_redirect_stdout()
    self.start = time()
    try:
      import heterocl as hcl
      self.hcl_print = hcl.print
    except:
      self.hcl_print = None

  def __del__(self):
    '''
    Destructor, clean up.
    '''
    if self.prof:
      self._save_profile()
    self.close()

  def __repr__(self):
    '''Representation as list of fields.'''
    s = ''
    for k, _ in TraceMgr._fields_:
      s += f'{k}: {getattr(self, k)}\n'
    return s[:-1]

  def _trace_init_remote(self, fd):
    pass

  def _trace_enable_remote(self, enable):
    pass

  def _trace_quit_remote(self):
    pass

  def _trace_flush_remote(self):
    pass

  def _trace_redirect_stdout_remote(self):
    pass

  def _enable_event_remote(self, key, val):
    # Should pass pair to remote, use local json for now.
    # Could upload json modes.
    pass

  def _collect_functions(self):
    if self.is_client:
      self.trace_init = self._trace_init_remote
      self.trace_enable = self._trace_enable_remote
      self.trace_quit = self._trace_quit_remote
      self.trace_flush = self._trace_flush_remote
      self.trace_redirect_stdout = self._trace_redirect_stdout_remote
    else:
      self.trace_init = gfunc('vta.runtime.trace.init')
      self.trace_enable = gfunc('vta.runtime.trace.enable')
      self.trace_quit = gfunc('vta.runtime.trace.quit')
      self.trace_flush = gfunc('vta.runtime.trace.flush')
      self.trace_redirect_stdout = gfunc('vta.runtime.trace.redirect_stdout')
    if self.target == 'fsim':
      self.prof_stats = gfunc('vta.simulator.profiler_status')
    elif self.target == 'tsim':
      self.prof_stats = gfunc('vta.tsim.profiler_status')
    elif self.target == 'de10nano':
      self.prof_stats = gfunc('vta.de10nano.profiler_status', True)
    else:
      self.prof_stats = None

  def _decode_mode(self):
    '''Set Trace Mode.'''
    fn = f'{home()}/test/trace_modes.json'
    modes = []
    fp = open(fn, 'r')
    modes = json.load(fp)
    fp.close()
    fl = f'{home()}/test/trace_modes_local.json'
    if os.path.exists(fl):
      fp = open(fl, 'r')
      local_modes = json.load(fp)
      modes.update(local_modes)
      fp.close()
    mode = modes.get(self.mode, {})
    fields = {k[0] for k in self._fields_}
    for key, val in mode.items():
      if key in fields:
        self.__setattr__(key, val)
    dump = mode.get("dump", {})
    for key, val in dump.items():
      if key in fields:
        self.__setattr__(key, val)
    event = mode.get("event", {})
    if self.is_client:
      self.enable_event = self._enable_event_remote
    else:
      self.enable_event = gfunc('vta.runtime.trace.enable_event')
    for key, val in event.items():
      self.enable_event(key, val)
      try:
        if val:
          self.m_enable_event.add(key)
        else:
          self.m_enable_event.remove(key)
      except: pass

  def valid (self):
    return self.trace and self.fd != -1

  def close(self):
    if self.valid():
      self.trace_quit()

  def init_remote_trace(self, remote):
    remote.get_function("tvm.contrib.vta.init_trace_mgr")(self.node, self.mode)

  def done_remote_trace(self, remote):
    remote.get_function("tvm.contrib.vta.done_trace_mgr")()
    modes = {self.mode, 'prof'} if self.prof else {self.mode}
    for mode in modes:
      user = 'xilinx' if self.target == 'pynq' else 'fpga'
      rpath = f'/home/{user}/tvm/verif/work/{self.node}.{mode}'
      lpath = trace_path(self.node, mode)
      with open(lpath, 'wb') as fp:
        if remote.exists(rpath):
          fp.write(remote.download(rpath))
    
  def _node_info(self):
    pos = self.node.rfind('[')
    test = self.node[0:pos]
    knob = self.node[pos+1:-1].replace(self.target,'').rstrip('-')
    return test, knob

  #def capture_log(self, cap):
  #  if self.log:
  #    with cap.disabled():
  #      capture = cap.readouterr()
  #      if capture.out:
  #        fn = trace_path(self.node, 'out')
  #        with open(fn, 'w') as fp:
  #          fp.write(capture.out)
  #      if capture.err:
  #        fn = trace_path(self.node, 'err')
  #        with open(fn, 'w') as fp:
  #          fp.write(capture.err)

  def _save_profile(self):
    fn = trace_path(self.node, 'prof')
    with open(fn, 'w') as fp:
      if self.prof_stats is not None:
        stats = json.loads(self.prof_stats())
      else:
        stats = {}
      stats['seconds'] = f'{time() - self.start:.2f}'
      stats['target'] = self.target
      stats['node'] = self.node
      stats['test'], stats['knob'] = self._node_info()
      json.dump(stats, fp, indent=2, sort_keys=True)

  def Enabled (self):
    return self.valid()

  def Header(self, event, format, data=[]):
    self.Event(event+':', format, data)

  def Event(self, event, format, data=[]):
    cont = event.startswith('+')
    if cont: event = event[1:]
    if self.target == 'bsim' and event in self.m_enable_event:
      if cont:
        self.hcl_print(data, format)
      else:
        self.hcl_print(data, f'{event:12s}{format}')


def trace_mgr(node=None, mode=None):
  '''
  Create a trace object and register it in builtins.
  By default the trace is not backed up by a trace file.
  A trace can be generated as work/<stem>.<mode>.
  Call after runtime library is loaded when trace is not
  managed by pytest also selecting a target with VTA_TARGET
  or vta_config.json.
  Example:
  <imports>
  from verif.trace_mgr import trace_mgr
  <body>
  trace_mgr()
  test()
  '''
  builtins.trace_mgr = TraceMgr(node, mode)
  return builtins.trace_mgr

@contextmanager
def config_trace(node, mode):
  '''
  Trace Configuration Context.
  Call after runtime library is loaded.
  It has got so simple that just an instance object context could do.
  '''
  try:
    # Share the trace_manager globally.
    builtins.trace_mgr = TraceMgr(node, mode)
    yield builtins.trace_mgr
  except:
    raise
  finally:
    #del tmgr
    builtins.trace_mgr = None
