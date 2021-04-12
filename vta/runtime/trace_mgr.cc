// Modified by contributors from Intel Labs

#include "trace_mgr.h"

namespace vta {

  TraceMgr trace_mgr;

  void * TraceMgr::Init (int fd) {
    m_fd = fd;
    if (fd != -1) {
      m_fp = fdopen(m_fd, "w");
      //RedirectStdout();
    }
    else
      m_fp = NULL;
    return this;
  }

  void TraceMgr::Quit () {
    if (m_fp) {
      fclose(m_fp);
      RestoreStdout();
      fflush(stdout);
    }
    m_fp = NULL;
    m_fd = -1;
  }

  void TraceMgr::Flush () {
    if (m_fp)
      fflush(m_fp);
  }

  void TraceMgr::RedirectStdout () {
    if (m_fp && !m_stdout) {
      m_stdout = stdout;
      stdout = m_fp;
    }
  }

  void TraceMgr::RestoreStdout () {
    if (m_stdout) {
      stdout = m_stdout;
      m_stdout = NULL;
    }
  }

  void TraceMgr::EnableEvent (const std::string &evt, bool enable) {
    if (enable)
      m_event_enable.insert(evt);
    else
      m_event_enable.erase(evt);
  }

  bool TraceMgr::EventEnabled (const std::string &evt) {
    return Trace() && m_event_enable.count(evt);
  }

  bool TraceMgr::EventEnabled (const char *evt) {
    return Trace() && m_event_enable.count(std::string(evt));
  }

  // Called with Verilator formatted output from Chisel printf.
  // Any call must start with the event name optionally prepended by a 
  // character. If the first character is ':' this is a header event,
  // if '+' a continuation event, otherwise the start of a event.
  void TraceMgr::VL_EVENT (const std::string &s) {
    if (m_fp) {
      bool cont = s[0] == '+';
      bool head = s[0] == ':';
      size_t beg = cont || head ? 1 : 0;
      size_t pos = s.find(' ');
      const std::string &event = s.substr(beg, pos-beg);
      if (EventEnabled(event) && (!head || EventEnabled("header"))) {
        if (!cont) {
          const char *fmt = head ? m_header_fmt : m_event_fmt;
          fprintf(m_fp, fmt, event.c_str());
        }
        fputs(s.substr(pos+1).c_str(), m_fp);
      }
    }
  }

  // Called by runtime and C++ models.
  void TraceMgr::printf (const char *format, ... ) {
    if (m_fp) {
      va_list args;
      va_start(args, format);
      if (m_enable_trace)
        vfprintf(m_fp, format, args);
      else
        vfprintf(stdout, format, args);
      va_end(args);
    }
  }

  void TraceMgr::Header (const char *evt, const char *fmt, ... ) {
    if (EventEnabled(evt) && EventEnabled("header")) {
      fprintf(m_fp, m_header_fmt, evt);
      va_list args;
      va_start(args, fmt);
      vfprintf(m_fp, fmt, args);
      va_end(args);
    }
  }

  void TraceMgr::Event (const char *evt, const char *fmt, ... ) {
    bool cont = *evt == '+';
    bool head = *evt == ':';
    if (cont || head)
      evt++;
    if (EventEnabled(evt) && (!head || EventEnabled("header"))) {
      if (!cont) {
        const char *fmt = head ? m_header_fmt : m_event_fmt;
        fprintf(m_fp, fmt, evt);
      }
      va_list args;
      va_start(args, fmt);
      vfprintf(m_fp, fmt, args);
      va_end(args);
    }
  }
}
