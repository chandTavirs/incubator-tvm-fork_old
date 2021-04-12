"""
Trace enable module including class and function versions.
"""

# Modified by contributors from Intel Labs

from . trace_mgr import trace_mgr, TraceMgr

class TraceEnable(TraceMgr):
  '''
  Trace Manager with enable settings.
  '''
  def __new__(cls):
    return super(TraceMgr, cls).__new__(cls)

  def __init__(self):
    TraceMgr.__init__(self)
    self.trace = True

def trace_enable():
  tmgr = trace_mgr()
  tmgr.trace = True
  return tmgr
