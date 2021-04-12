/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Modified by contributors from Intel Labs

/*!
 * \file trace_mgr.h
 * \brief VTA trace manager.
 */

#ifndef VTA_RUNTIME_TRACEMGR_H_
#define VTA_RUNTIME_TRACEMGR_H_

#define VL_FWRITEF vta::trace_mgr.fwritef

#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <set>
#include <string>

namespace vta {
// Passing a class instance by pointer works and unpacked
// properly in python. It would probably not work in the presence
// of a virtual pointer though. Methods need to be static to match
// C like calling conventions. Any memory allocation needs to be done
// carefully and owned in one place.
// Move implementation in cc file later.
struct TraceMgr {
  // Trace starts with a valid file descriptor and a buffered
  // stream is created to dump data throughout. The stream is
  // synchronized and flushed in the code accordingly among
  // different threads. The file descriptor may be used in
  // the python side to append bulk data at the end before quit
  // is called.
  void * Init (int fd);
  void Quit ();
  void Flush ();
  void RedirectStdout ();
  void RestoreStdout ();
  FILE * Fp () { return m_fp; }

  void Trace (bool enable) { m_enable_trace = enable; }
  bool Trace          () { return m_enable_trace; }
  bool TraceHost      () { return m_enable_trace && m_enable_host; }
  bool TraceAlloc     () { return m_enable_trace && m_enable_alloc; }
  bool TraceLoad      () { return m_enable_trace && m_enable_load; }
  bool TraceStore     () { return m_enable_trace && m_enable_store; }
  bool TraceInsn      () { return m_enable_trace && m_enable_insn; }
  bool Profile        () { return m_enable_prof; }
  bool TraceVirtAddr  () { return m_enable_trace && m_enable_virt; }
  bool TraceXY        () { return m_enable_trace && m_enable_xy; }
  bool TraceGemm      () { return m_enable_trace; }

  bool EventEnabled (const char *evt);
  bool EventEnabled (const std::string &evt);
  void EnableEvent (const std::string &evt, bool enable);
  
  // General purpose printf replacement to redirect stdout to trace.
  void printf (const char *format, ... );
  // Trace event redirection from from Chisel through Verilator.
  void VL_EVENT (const std::string &s);
  // Trace event header for C++ models.
  void Header (const char *evt, const char *fmt, ...);
  // Trace event data for C++ models.
  void Event (const char *evt, const char *fmt, ...);

private:

  // These are the members to match in the corresponding ctypes
  // structure in trace_mgr.py, in order.
  int m_fd = -1;
  bool m_enable_trace = false;
  bool m_enable_host = false;
  bool m_enable_alloc = false;
  bool m_enable_load = false;
  bool m_enable_store = false;
  bool m_enable_insn = false;
  bool m_enable_prof = false;
  bool m_enable_xy = false;
  bool m_enable_virt = false;

private:
 
  // Do not export below mebers.
  FILE *m_fp = NULL;
  FILE *m_stdout = NULL;

  const char *m_event_fmt  = "%-12s";
  const char *m_header_fmt = ":%-12s";
  std::set<std::string> m_event_enable;
};

extern TraceMgr trace_mgr;

}
#endif  // VTA_RUNTIME_TRACEMGR_H_
