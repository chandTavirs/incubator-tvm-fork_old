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
 * \file runtime.cc
 * \brief Generic VTA runtime in C++11.
 *
 *  The runtime depends on specific instruction
 *  stream spec as specified in hw_spec.h
 */
#include "runtime.h"

#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include <vta/driver.h>
#include <vta/hw_spec.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

#include "trace_mgr.h"

#ifdef printf
#undef printf
#endif  // printf
#define printf trace_mgr.printf

const char* mem_fmt = "%-16s virt_addr: 0x%016" PRIxPTR
                      " phys_addr: 0x%08x "
                      "sram_addr: 0x%08x elem_bytes: %4" PRIu32 " size_bytes: %8" PRIu32 "\n";
const char* host_fmt = "%-16s  dst_addr: 0x%016" PRIxPTR
                       "  src_addr: 0x%08x "
                       "sram_addr: 0x%08x elem_bytes: %4" PRIu32 " size_bytes: %8" PRIu32 "\n";
namespace vta {

// Avoid bad configurations.
static_assert(VTA_UOP_WIDTH == sizeof(VTAUop) * 8, "VTA_UOP_WIDTH does not match VTAUop size");

/*! \brief Enable coherent access of data buffers between VTA and CPU */
static const bool kBufferCoherent = VTA_COHERENT_ACCESSES;
/*! \brief Always cache buffers (otherwise, write back to DRAM from CPU) */
static const bool kAlwaysCache = true;

/*!
 * \brief Data buffer represents data on CMA.
 */
struct DataBuffer {
  /*! \return Virtual address of the data. */
  void* virt_addr() const { return data_; }
  /*! \return Physical address of the data. */
  vta_phy_addr_t phy_addr() const { return phy_addr_; }
  /*!
   * \brief Invalidate the cache of given location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void InvalidateCache(size_t offset, size_t size) {
    if (!kBufferCoherent && kAlwaysCache) {
      VTAInvalidateCache(reinterpret_cast<char*>(data_) + offset, phy_addr_ + offset, size);
    }
  }
  /*!
   * \brief Invalidate the cache of certain location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void FlushCache(size_t offset, size_t size) {
    if (!kBufferCoherent && kAlwaysCache) {
      VTAFlushCache(reinterpret_cast<char*>(data_) + offset, phy_addr_ + offset, size);
    }
  }
  /*!
   * \brief Performs a copy operation from host memory to buffer allocated with VTAMemAlloc.
   * \param dst The destination buffer in FPGA-accessible memory. Has to be allocated with
   * VTAMemAlloc(). \param src The source buffer in host memory. \param size Size of the region in
   * Bytes.
   */
  void MemCopyFromHost(void* dst, const void* src, size_t size) {
    if (trace_mgr.TraceHost() && trace_mgr.TraceVirtAddr())
      printf(host_fmt, "CopyFromHost", dst, src, NULL, 0, size);
    VTAMemCopyFromHost(dst, src, size);
  }
  /*!
   * \brief Performs a copy operation from buffer allocated with VTAMemAlloc to host memory.
   * \param dst The destination buffer in host memory.
   * \param src The source buffer in FPGA-accessible memory. Has to be allocated with VTAMemAlloc().
   * \param size Size of the region in Bytes.
   */
  void MemCopyToHost(void* dst, const void* src, size_t size) {
    if (trace_mgr.TraceHost() && trace_mgr.TraceVirtAddr())
      printf(host_fmt, "CopyToHost", dst, src, NULL, 0, size);
    VTAMemCopyToHost(dst, src, size);
  }
  /*!
   * \brief Allocate a buffer of a given size.
   * \param size The size of the buffer.
   */
  static DataBuffer* Alloc(size_t size) {
    void* data = VTAMemAlloc(size, kAlwaysCache);
    CHECK(data != nullptr);
    DataBuffer* buffer = new DataBuffer();
    buffer->data_ = data;
    buffer->phy_addr_ = VTAMemGetPhyAddr(data);
    if (trace_mgr.TraceAlloc()) {
      printf(mem_fmt, "BufferAlloc", data, buffer->phy_addr_, NULL, 0, size);
    }
    garbage_collector_.insert(buffer);
    return buffer;
  }
  /*!
   * \brief Free the data buffer.
   * \param buffer The buffer to be freed.
   */
  static void Free(DataBuffer* buffer) {
    VTAMemFree(buffer->data_);
    if (trace_mgr.TraceAlloc()) {
      printf(mem_fmt, "BufferFree", buffer->data_, NULL, NULL, 0, 0);
    }
    garbage_collector_.erase(buffer);
    delete buffer;
  }
  /*!
   * \brief Create data buffer header from buffer ptr.
   * \param buffer The buffer pointer.
   * \return The corresponding data buffer header.
   */
  static DataBuffer* FromHandle(const void* buffer) {
    return const_cast<DataBuffer*>(reinterpret_cast<const DataBuffer*>(buffer));
  }

  static void Collect () {
    std::set<void*>::iterator i, e = garbage_collector_.end();
    for (i = garbage_collector_.begin(); i != e; i++) {
      auto buffer = (DataBuffer*) *i;
      if (trace_mgr.TraceAlloc()) {
        printf(mem_fmt, "BufferCollect", buffer->data_, NULL, NULL, 0, 0);
      }
      VTAMemFree(buffer->data_);
      delete buffer;
    }
    garbage_collector_.clear();
  }

 private:
  /*! \brief The internal data. */
  void* data_;
  /*! \brief The physical address of the buffer, excluding header. */
  vta_phy_addr_t phy_addr_;

  static std::set<void*> garbage_collector_;
};

std::set<void*> DataBuffer::garbage_collector_;

/*!
 * \brief Micro op kernel.
 *  Contains functions to construct the kernel with prefix Push.
 */
class UopKernel {
 public:
  /*! \brief Loop information. */
  struct LoopEntry {
    uint32_t extent;
    uint32_t dst_factor;
    uint32_t src_factor;
    uint32_t wgt_factor;
  };
  /*!
   * \brief Construct UopKernel with signature.
   * \param signature The pointer to signature.
   * \param nbytes Number of bytes.
   */
  UopKernel(const char* signature, int nbytes) : signature_(signature, signature + nbytes) {}
  /*!
   * \brief Verify if the signature is correct.
   * \param signature Signature ptr.
   * \param nbytes Number of bytes.
   */
  bool MatchSignature(void* signature, int nbytes) const {
    if (static_cast<size_t>(nbytes) != signature_.size()) return false;
    return memcmp(signature, signature_.data(), nbytes) == 0;
  }
  /*! \return Whether the kernel is cached in SRAM. */
  bool cached() const { return sram_begin_ != sram_end_; }
  /*! \return The length of the micro op sequence. */
  size_t size() const { return seq_.size(); }
  /*! \return The micro-op data. */
  const VTAUop* data() const { return seq_.data(); }
  /*! \return The loop structure. */
  const std::vector<LoopEntry>& loop() const { return loop_; }
  /*!
   * \brief Declare loop start.
   * \param extent The loop extent.
   * \param dst_factor Loop factor of accum index.
   * \param src_factor Loop factor of input index
   * \param wgt_factor Loop factor of weight index.
   */
  void PushLoopBegin(uint32_t extent, uint32_t dst_factor, uint32_t src_factor,
                     uint32_t wgt_factor) {
    LoopEntry le;
    le.extent = extent;
    le.dst_factor = dst_factor;
    le.src_factor = src_factor;
    le.wgt_factor = wgt_factor;
    CHECK_EQ(seq_.size(), 0U);
    CHECK_LT(loop_.size(), 2U);
    loop_.push_back(le);
    ++loop_ptr_;
  }
  /*!
   * \brief Declare loop end.
   */
  void PushLoopEnd() { --loop_ptr_; }
  /*!
   * \brief Push micro op into kernel.
   * \param mode Set to GEMM mode if set to 0, ALU mode is set to 1.
   * \param reset_out Resets the accum to 0.
   * \param dst_index The accum memory index.
   * \param src_index The input memory (gemm) / accum memory (alu) index.
   * \param wgt_index The weight memory index.
   * \param opcode The ALU opcode.
   * \param use_imm Use immediate in ALU mode if set to true.
   * \param imm_val Immediate value in ALU mode.
   */
  void Push(uint32_t mode, uint32_t reset_out, uint32_t dst_index, uint32_t src_index,
            uint32_t wgt_index, uint32_t opcode, uint32_t use_imm, int32_t imm_val) {
    // The loop nest structure
    // VerifyDep(dst_index);
    VTAUop op;
    op.alu.dst_idx = dst_index;
    if (mode == 1) { // ALU mode: put lower bits in src_idx and any remaining bits in wgt_idx
      op.alu.src_idx = src_index;
      CHECK(op.alu.src_idx == src_index);
    } else { // keep other modes unchanged
      op.gem.src_idx = src_index;
      op.gem.wgt_idx = wgt_index;
      CHECK(op.gem.src_idx == src_index);
      CHECK(op.gem.wgt_idx == wgt_index);
    }
    seq_.push_back(op);
    // Ensure that mode is consistent if set
    if (mode_ == 0xFFFFFFFF) {
      mode_ = mode;
    } else {
      CHECK(mode_ == mode);
    }
    // Set reset_out field if unset
    if (reset_out_ == 0xFFFFFFFF) {
      reset_out_ = reset_out;
    } else {
      CHECK(reset_out_ == reset_out);
    }
    // Check kernel op and imm/imm_val in ALU mode
    if (mode == 1) {
      if (opcode_ == 0xFFFFFFFF) {
        opcode_ = opcode;
        use_imm_ = use_imm;
        imm_val_ = imm_val;
      } else {
        CHECK(opcode_ == opcode);
        CHECK(use_imm_ == use_imm);
        CHECK(imm_val_ == imm_val);
      }
    }
  }
  /*! \brief Dump kernel micro ops to stdout. */
  void Dump() {
    uint32_t size = seq_.size();
    printf("There are %u uops [%u, %u]\n", size, sram_begin_, sram_end_);
    for (uint32_t i = 0; i < size; ++i) {
      printf("[%04u]\t acc=%u, inp=%u, wgt=%u\n", i, seq_[i].gem.dst_idx, seq_[i].gem.src_idx,
             seq_[i].gem.wgt_idx);
    }
    printf("\n");
  }

 public:
  // The kernel's mode, opcode, immediate setting and value
  uint32_t mode_{0xFFFFFFFF};  // UOP type: 0xFFFFFFFF - unset, 0 - GEMM, 1 - ALU
  uint32_t opcode_{0xFFFFFFFF};
  uint32_t reset_out_{0xFFFFFFFF};
  bool use_imm_{false};
  int16_t imm_val_{0};

 private:
  // Verify that we don't write to the same acc_mem index two cycles in a row
  void VerifyDep(uint32_t dst_index) {
    size_t step = std::min(static_cast<size_t>(2U), seq_.size());
    for (size_t i = seq_.size() - step; i < seq_.size(); ++i) {
      CHECK(seq_[i].gem.dst_idx != dst_index);
    }
  }
  // The uop buffer
  template <int, bool, bool>
  friend class UopQueue;
  friend class CommandQueue;
  // SRAM location if begin != end
  uint32_t sram_begin_{0};
  uint32_t sram_end_{0};
  // The signature used for verification
  std::vector<char> signature_;
  // Internal sequence
  std::vector<VTAUop> seq_;
  // The loop nest structure specific to ALU instructions
  std::vector<LoopEntry> loop_;
  // The loop pointer
  size_t loop_ptr_{0};
};

/*!
 * \brief Base class of all queues to send and recv serial data.
 */
template <class T>
class BaseQueue {
 public:
  void FreeSpace () {
    if (fpga_buff_ != nullptr) {
      if (trace_mgr.TraceAlloc()) {
        printf(mem_fmt, "QueueFree", fpga_buff_, NULL, NULL, 0, 0);
      }
      VTAMemFree(fpga_buff_);
      fpga_buff_ = nullptr;
    }
  }
  virtual ~BaseQueue() {
    if (fpga_buff_ != nullptr)
      FreeSpace();
  }
  /*! \return Content of DRAM buffer. */
  char* dram_buffer() const { return dram_buffer_; }
  /*! \return Physical address of DRAM. */
  vta_phy_addr_t dram_phy_addr() const {
    CHECK(fpga_buff_phy_);
    return fpga_buff_phy_;
  }
  /*! \return Whether there is pending information. */
  bool pending() const { return sram_begin_ != sram_end_; }
  /*! \brief Initialize the space of the buffer. */
  void InitSpace(uint32_t elem_bytes, uint32_t max_bytes, bool coherent, bool always_cache) {
    coherent_ = coherent;
    always_cache_ = always_cache;
    elem_bytes_ = elem_bytes;
    // Allocate buffer ahead of time
    fpga_buff_ = static_cast<char*>(VTAMemAlloc(max_bytes, coherent_ || always_cache_));
    CHECK(fpga_buff_ != nullptr);
    fpga_buff_phy_ = VTAMemGetPhyAddr(fpga_buff_);
  }
  /*!
   * \brief Reset the pointer of the buffer.
   *  Set SRAM pointer to be the current end.
   */
  virtual void Reset() {
    dram_buffer_.clear();
    // reset to 0 as we always copy data to area starting from fpga_buff base
    // we do mem copy for every DeviceRun
    sram_end_ = 0;
    sram_begin_ = sram_end_;
  }

 protected:
  // Cache coherence access (shared memory only)
  bool coherent_{false};
  // Make the buffer cacheable
  bool always_cache_{false};
  // Element bytes
  uint32_t elem_bytes_{0};
  // Begin location of current SRAM read in FIFO mode
  uint32_t sram_begin_{0};
  // End location of current SRAM write in FIFO mode
  uint32_t sram_end_{0};
  // The buffer in DRAM
  std::vector<T> dram_buffer_;
  // FPGA accessible buffer
  void* fpga_buff_{NULL};
  // Physical address of the FPGA buffer
  vta_phy_addr_t fpga_buff_phy_{0};
};

// forward declaration
template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class InsnQueue;

/*!
 * \brief Micro op buffer that manages the micro op cache.
 */
template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class UopQueue : public BaseQueue<UopKernel *> {
 public:
  void InitSpace() {
    BaseQueue::InitSpace(kElemBytes, kMaxBytes, kCoherent, kAlwaysCache);
    if (trace_mgr.TraceAlloc()) {
      printf(mem_fmt, "UopQueue", fpga_buff_, fpga_buff_phy_, NULL, kElemBytes, kMaxBytes);
    }
  }
  // Push data to the queue
  template <typename FAutoSync>
  void Push(UopKernel* kernel, FAutoSync fautosync) {
    // if kernel is already cached in VTA SRAM, must already be in DRAM too, so skip further processing
    if (kernel->cached()) return;
    // check if we've exceeded the size of the allocated FPGA readable buffer
    size_t num_op = kernel->size();
    // make sure this kernel can fit in SRAM (by itself)
    CHECK(num_op <= kMaxNumUop); // Cannot have a single kernel larger than SRAM buffer
    CHECK(num_op > 0); // some ops to add
    if (VTA_ENABLE_RUNTIME_VALUE_BASED_UOP_CONSOLIDATION) { // are kernel uops already in SRAM?
      if (!pending() && cache_idx_ > 0) { // a kernel was written before
        UopKernel * prev = cache_[cache_idx_ - 1];
        CHECK(prev);
        if (prev->cached() && num_op == prev->size()) { // prev kernel is cached and same size
          bool all_match = true; // assume everything matches to start
          for (uint32_t i = 0; i < num_op; i++) { // go through all uops
            const VTAUop prev_uop = prev->data()[i];
            const VTAUop cur_uop = kernel->data()[i];

            if (prev_uop.gem.dst_idx != cur_uop.gem.dst_idx || 
                prev_uop.gem.wgt_idx != cur_uop.gem.wgt_idx ||
                prev_uop.gem.src_idx != cur_uop.gem.src_idx) { // any mismatch?
              all_match = false;
              break; // no need to look at remaining uops: one mismatch is enough
            }
          }
          // all uops match
          if (all_match) { // copy from predecessor
            kernel->sram_begin_ = prev->sram_begin_;
            kernel->sram_end_ = prev->sram_end_;
            CHECK(kernel->cached());
            cache_.insert(cache_.begin() + cache_idx_, kernel);
            cache_idx_ += 1;
            return;
          } // else fall through and eventually create new uop(s) for SRAM
        }
      }
    }
    // at this point, kernel is not already in SRAM and is not a duplicate of previous kernel
    bool found_in_dram = false; // is it already in DRAM?
    size_t ops_already_in_dram = 0; // also calculate how many ops are in DRAM right now
    for (uint32_t i = 0; i < dram_buffer_.size(); ++i) {
      if (kernel == dram_buffer_[i]) {
        found_in_dram = true;
        break;
      }
      ops_already_in_dram += dram_buffer_[i]->size();
    }

    if (!found_in_dram) { // not already in dram, so add it
      dram_buffer_.push_back(kernel);
      // check if we've exceeded the size of the allocated FPGA readable buffer
      if (ops_already_in_dram + num_op > kMaxElems) {
        fautosync(); // make more room in the fpga-accessible dram buffer
        CHECK(ops_already_in_dram + num_op <= kMaxElems); // enough room now
      }
    }

    // now try to fit this kernel into the SRAM
    uint32_t uop_begin = 0;
    if (sram_end_ + num_op > kMaxNumUop) { // Need to evict
      cache_idx_ = 0;
      sram_begin_ = 0;
      sram_end_ = num_op;
    } else { // just add to the end
      uop_begin = sram_end_;
      sram_end_ += num_op;
    }

    // Simple eviction policy
    uint32_t evict_begin = cache_idx_;
    for (; cache_idx_ < cache_.size(); ++cache_idx_) {
      if (cache_[cache_idx_]->sram_begin_ >= sram_end_) break; // not cached already
      // Mark the kernel as "invalid"
      cache_[cache_idx_]->sram_begin_ = 0;
      cache_[cache_idx_]->sram_end_ = 0;
    }
    // Increase size of buffer
    kernel->sram_begin_ = uop_begin;
    kernel->sram_end_ = sram_end_;
    CHECK(kernel->cached());
    cache_.insert(cache_.begin() + cache_idx_, kernel);
    cache_.erase(cache_.begin() + evict_begin, cache_.begin() + cache_idx_);
    cache_idx_ = evict_begin + 1;
  }

  // Flush micro op load instruction: opportunistically combine with prior loads
  void FlushUopLoad(InsnQueue<kMaxBytes, kCoherent, kAlwaysCache> & insn_q) {
    CHECK(pending()); // no need to create a load uop insn if no kernel
    CHECK(cache_idx_ > 0); // can't flush a uop when there's nothing in SRAM
    UopKernel * cur = cache_[cache_idx_ - 1];
    CHECK(cur);
    CHECK(cur->size() == (sram_end_ - sram_begin_));

    // Derive offset address in FPGA-readable buffer
    uint32_t offset_bytes = 0;
    for (uint32_t i = 0; i < dram_buffer_.size(); ++i) {
      if (dram_buffer_[i] == cur) {
        break;
      }
      offset_bytes += dram_buffer_[i]->size() * kElemBytes;
    }
    uint32_t effective_dram_base = (fpga_buff_phy_ + offset_bytes) / kElemBytes;

    if (VTA_ENABLE_RUNTIME_GREEDY_UOP_LOADS) {
      //if possible, find a previous load uop instruction to extend
      union VTAInsn c;
      for (int32_t i = insn_q.count() - 1; i >= 0; i--) { // find the most recent insn
        c.generic = insn_q.data()[i];
        if (c.mem.opcode == VTA_OPCODE_LOAD && c.mem.memory_type == VTA_MEM_ID_UOP) { // uop load
          if (c.mem.sram_base + c.mem.x_size == sram_begin_ && // adjacent sram
              c.mem.dram_base + c.mem.x_size == effective_dram_base && // adjacent dram
              cache_idx_ > 1 && // previous kernel exists
              cache_[cache_idx_-1]->sram_end_ > cache_[cache_idx_-1]->sram_begin_  && // in SRAM
              c.mem.sram_base + c.mem.x_size + (sram_end_ - sram_begin_) < kMaxNumUop // fits
          ) {
            // just extend that previous instruction's x_size and x_stride
            reinterpret_cast<VTAMemInsn*>(insn_q.data())[i].x_size += (sram_end_ - sram_begin_);
            reinterpret_cast<VTAMemInsn*>(insn_q.data())[i].x_stride += (sram_end_ - sram_begin_);
            // Reset indices
            sram_begin_ = sram_end_; // nothing is pending anymore
            return;
          } else {
            break; // do not consider any other instructions
          }
        }
      } // if we get here, coudn't find an existing load uop to modify, so we create a new one
    }

    VTAMemInsn* insn = insn_q.CreateMemInsn(VTA_MEM_ID_UOP);
    CHECK(insn);
    insn->opcode = VTA_OPCODE_LOAD;
    insn->memory_type = VTA_MEM_ID_UOP;
    insn->sram_base = sram_begin_;
    insn->dram_base = effective_dram_base;
    insn->y_size = 1;
    insn->x_size = (sram_end_ - sram_begin_);
    insn->x_stride = (sram_end_ - sram_begin_);
    insn->y_pad_0 = 0;
    insn->y_pad_1 = 0;
    insn->x_pad_0 = 0;
    insn->x_pad_1 = 0;
    // Reset indices
    sram_begin_ = sram_end_; // nothing is pending
  }

  /*! \brief clear cache and reset base queue buffer.*/
  void Reset() {
    // unmark "cached" status
    // as we cannot assume it is still in SRAM across DeviceRun
    for (UopKernel* kernel : cache_) {
      kernel->sram_begin_ = 0;
      kernel->sram_end_ = 0;
    }

    cache_.clear();
    cache_idx_ = 0;
    BaseQueue<UopKernel *>::Reset();
  }
  void AutoReadBarrier() { ReadBarrier(); }
  /*! \brief Writer barrier to make sure that data written by CPU is visible to VTA. */
  void ReadBarrier() {
    CHECK(fpga_buff_ != nullptr);
    CHECK(fpga_buff_phy_);
    // Iterate over caches; allocate buffer in FPGA-readable memory
    uint32_t buff_size = 0;
    for (uint32_t i = 0; i < dram_buffer_.size(); ++i) {
      buff_size += dram_buffer_[i]->size() * kElemBytes;
    }
    CHECK(buff_size <= kMaxBytes);
    // Move kernel contents to FPGA readable buffer
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dram_buffer_.size(); ++i) {
      uint32_t ksize = dram_buffer_[i]->size() * kElemBytes;
      VTAMemCopyFromHost(static_cast<char*>(fpga_buff_) + offset, dram_buffer_[i]->data(), ksize);
      // Update offset
      offset += ksize;
    }
    // Flush if we're using a shared memory system
    // and if interface is non-coherent
    if (!coherent_ && always_cache_) {
      VTAFlushCache(fpga_buff_, fpga_buff_phy_, offset);
    }
  }

  void DumpUop () {
    printf("There are %u uop kernels\n", cache_.size());
    for (UopKernel *kernel : cache_) {
      kernel->Dump();
    }
  }

 private:
  // Cache pointer
  uint32_t cache_idx_{0};
  // Cached ring, sorted by sram_begin
  std::vector<UopKernel*> cache_;
  // Constants
  static constexpr int kElemBytes = sizeof(VTAUop);
  static constexpr int kMaxNumUop = VTA_UOP_BUFF_DEPTH;
  static constexpr int kMaxElems = kMaxBytes / kElemBytes;
};

// Internal kernel structure
class UopKernelMap {
 public:
  // Simple hash map
  UopKernel** Get(void* signature, int nbytes) {
    uint64_t rowkey = 0;
    uint32_t colkey = 0;
    CHECK(nbytes == 0 || nbytes == sizeof(int) || nbytes == 2*sizeof(int));
    if (nbytes == sizeof(int)) {
      memcpy(&rowkey, signature, sizeof(int));
      rowkey = rowkey + 1;
    }
    if (nbytes == 2*sizeof(int)) {
      memcpy(&rowkey, signature, 2*sizeof(int));
      colkey = static_cast<uint32_t>(rowkey);
      rowkey = rowkey >> 32;
      colkey = colkey + 1;
      rowkey = rowkey + 1;
    }
    CHECK_LT(rowkey, 1000);
    CHECK_LT(colkey, 1000);
    if (kmap_.size() <= rowkey) {
      kmap_.resize(rowkey + 1);
    }
    if (kmap_[rowkey].size() <= colkey) {
      kmap_[rowkey].resize(colkey + 1, nullptr);
    }
    return &(kmap_[rowkey][colkey]);
  }

 private:
  std::vector<std::vector<UopKernel*>> kmap_;
};

enum PipelineStage : int { kNoneStage = 0, kLoadStage = 1, kComputeStage = 2, kStoreStage = 3 };

// Instruction Queue
template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class InsnQueue : public BaseQueue<VTAGenericInsn> {
 public:
  /*! \brief Initialize the space. */
  void InitSpace() {
    BaseQueue::InitSpace(kElemBytes, kMaxBytes, kCoherent, kAlwaysCache);
    // Initialize the stage
    std::fill(pending_pop_prev_, pending_pop_prev_ + 4, 0);
    std::fill(pending_pop_next_, pending_pop_next_ + 4, 0);
    if (trace_mgr.TraceAlloc()) {
      void* virt_addr = trace_mgr.TraceVirtAddr() ? fpga_buff_ : NULL;
      printf(mem_fmt, "InsnQueue", virt_addr, fpga_buff_phy_, NULL, kElemBytes, kMaxBytes);
    }
  }
  /*! \return The data pointer. */
  VTAGenericInsn* data() { return dram_buffer_.data(); }
  /*! \return Number of instructions. */
  uint32_t count() { return dram_buffer_.size(); }
  // Insert dependency push of load
  void DepPop(int from, int to) {
    // NOTE: This instruction executes on queue[to]
    if (from < to) {
      if (pending_pop_prev_[to]) {
        this->CommitPendingPop(to);
      }
      pending_pop_prev_[to] = 1;
    } else {
      if (pending_pop_next_[to]) {
        this->CommitPendingPop(to);
      }
      pending_pop_next_[to] = 1;
    }
    // Impossible condition
    CHECK(from != kLoadStage || to != kStoreStage);
    CHECK(from != kStoreStage || to != kLoadStage);
  }
  // Insert dependency push of load
  void DepPush(int from, int to) {
    // NOTE: this instruction executes on queue[from]
    this->CommitPendingPop(from);
    if (!dram_buffer_.empty()) {
      VTAMemInsn* mptr = reinterpret_cast<VTAMemInsn*>(&dram_buffer_.back());
      if (GetPipelineStage(mptr) == from) {
        if (from < to && !mptr->push_next_dep) {
          // push(LD->C) or push(C->ST)
          mptr->push_next_dep = true;
          return;
        } else if (from > to && !mptr->push_prev_dep) {
          // push(C->LD) or push(ST->C)
          mptr->push_prev_dep = true;
          return;
        }
      }
    }
    if (from < to) {
      // Push next dep
      PushNoop(from, false, true, false, false);
    } else {
      // Push prev dep
      PushNoop(from, true, false, false, false);
    }
  }
  // Create a new instruction for a GEMM stage
  VTAGemInsn* CreateGemInsn() { return reinterpret_cast<VTAGemInsn*>(Create(kComputeStage)); }
  // Create a new instruction for a ALU stage
  VTAAluInsn* CreateAluInsn() { return reinterpret_cast<VTAAluInsn*>(Create(kComputeStage)); }
  // Create a new instruction for a memory stage
  VTAMemInsn* CreateMemInsn(int memory_type) {
    return reinterpret_cast<VTAMemInsn*>(Create(GetMemPipelineStage(memory_type)));
  }
  // create a new instruction for a store stage
  VTAMemInsn* CreateStoreInsn() { return reinterpret_cast<VTAMemInsn*>(Create(kStoreStage)); }
  // Rewrite instruction stream to force serial execution
  void RewriteForceSerial() {
    int insn_count = count();
    VTAMemInsn* mem_ptr = reinterpret_cast<VTAMemInsn*>(data());
    VTAMemInsn* mem_last_store_ptr = nullptr;
    VTAMemInsn* mem_last_ptr = nullptr;
    for (int i = 1; i < insn_count; ++i) {
      PipelineStage prev = GetPipelineStageAll(mem_ptr + i - 1);
      PipelineStage now = GetPipelineStageAll(mem_ptr + i);
      if (prev == kLoadStage && now == kComputeStage) {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = true;
        mem_ptr[i].pop_prev_dep = true;
        mem_ptr[i].pop_next_dep = false;
      } else if (prev == kComputeStage && now == kLoadStage) {
        mem_ptr[i - 1].push_prev_dep = true;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = true;
      } else if (prev == kStoreStage && now == kComputeStage) {
        mem_ptr[i - 1].push_prev_dep = true;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = true;
      } else if (prev == kComputeStage && now == kStoreStage) {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = true;
        mem_ptr[i].pop_prev_dep = true;
        mem_ptr[i].pop_next_dep = false;
      } else {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = false;
      }
      if (now == kStoreStage) {
        mem_last_store_ptr = &mem_ptr[i];
      }
      mem_last_ptr = &mem_ptr[i];
    }
    // set dependency to make sure all core instruction get excuted
    // before last FINISH instruction
    if (mem_last_store_ptr && mem_last_ptr == mem_last_store_ptr) {
      mem_last_store_ptr->push_prev_dep = true;
      if (!pending_pop_next_[kComputeStage]) {
        DepPop(kStoreStage, kComputeStage);
      }
      CommitPendingPop(kComputeStage);
    } else {
        pending_pop_next_[kComputeStage] = 0;
    }
    DepPush(kComputeStage, kLoadStage);
    DepPop(kLoadStage, kComputeStage);
    if (!pending_pop_next_[kLoadStage]) {
      DepPop(kComputeStage, kLoadStage);
    }
    CommitPendingPop(kLoadStage);
    DepPush(kLoadStage, kComputeStage);
    CommitPendingPop(kComputeStage);
  }
  // Helper function: Get Opcode string
  const char* getOpcodeString(int opcode, bool use_imm) {
    // The string name
    if (opcode == VTA_ALU_OPCODE_MIN) {
      if (use_imm) {
        return "min imm";
      } else {
        return "min";
      }
    } else if (opcode == VTA_ALU_OPCODE_MAX) {
      if (use_imm) {
        return "max imm";
      } else {
        return "max";
      }
    } else if (opcode == VTA_ALU_OPCODE_ADD) {
      if (use_imm) {
        return "add imm";
      } else {
        return "add";
      }
    } else if (opcode == VTA_ALU_OPCODE_SHR) {
      return "shr";
    } else if (opcode == VTA_ALU_OPCODE_CLP) {
      return "clp";
    }
    else if (opcode == VTA_ALU_OPCODE_MOV) {
      if (use_imm) {
        return "mov imm";
      } else {
        return "mov";
      }
    }

    return "unknown op";
  }
  // Dump instructions in the queue
  void DumpInsn() {
    // Keep tabs on dependence queues
    int l2g_queue = 0;
    int g2l_queue = 0;
    int s2g_queue = 0;
    int g2s_queue = 0;
    // Converter
    union VTAInsn c;
    // Iterate over all instructions
    int insn_count = count();
    const VTAGenericInsn* insn = data();
    printf("There are %u instructions\n", insn_count);
    for (int i = 0; i < insn_count; ++i) {
      // Fetch instruction and decode opcode
      c.generic = insn[i];
      printf("INSTRUCTION %u: ", i);
      if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.x_size == 0) {
          if (c.mem.opcode == VTA_OPCODE_STORE) {
            printf("NOP-STORE-STAGE\n");
          } else if (GetMemPipelineStage(c.mem.memory_type) == kComputeStage) {
            printf("NOP-COMPUTE-STAGE\n");
          } else {
            printf("NOP-MEMORY-STAGE\n");
          }
          printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                 static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                 static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
          // Count status in queues
          if (c.mem.opcode == VTA_OPCODE_STORE) {
            CHECK(c.mem.pop_next_dep == false);
            CHECK(c.mem.push_next_dep == false);
            if (c.mem.pop_prev_dep) g2s_queue--;
            if (c.mem.push_prev_dep) s2g_queue++;
          } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                     (c.mem.memory_type == VTA_MEM_ID_INP || c.mem.memory_type == VTA_MEM_ID_WGT)) {
            CHECK(c.mem.pop_prev_dep == false);
            CHECK(c.mem.push_prev_dep == false);
            if (c.mem.pop_next_dep) g2l_queue--;
            if (c.mem.push_next_dep) l2g_queue++;
          } else {
            if (c.mem.pop_prev_dep) l2g_queue--;
            if (c.mem.push_prev_dep) g2l_queue++;
            if (c.mem.pop_next_dep) s2g_queue--;
            if (c.mem.push_next_dep) g2s_queue++;
          }
          printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
          printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
          continue;
        }
        // Print instruction field information
        if (c.mem.opcode == VTA_OPCODE_LOAD) {
          printf("LOAD ");
          if (c.mem.memory_type == VTA_MEM_ID_UOP) printf("UOP\n");
          if (c.mem.memory_type == VTA_MEM_ID_WGT) printf("WGT\n");
          if (c.mem.memory_type == VTA_MEM_ID_INP) printf("INP\n");
          if (c.mem.memory_type == VTA_MEM_ID_ACC) printf("ACC\n");
        }
        if (c.mem.opcode == VTA_OPCODE_STORE) {
          printf("STORE:\n");
        }
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        printf("\tDRAM: 0x%08x, SRAM:0x%04x\n", static_cast<int>(c.mem.dram_base),
               static_cast<int>(c.mem.sram_base));
        printf("\ty: size=%d, pad=[%d, %d]\n", static_cast<int>(c.mem.y_size),
               static_cast<int>(c.mem.y_pad_0), static_cast<int>(c.mem.y_pad_1));
        printf("\tx: size=%d, stride=%d, pad=[%d, %d]\n", static_cast<int>(c.mem.x_size),
               static_cast<int>(c.mem.x_stride), static_cast<int>(c.mem.x_pad_0),
               static_cast<int>(c.mem.x_pad_1));
      } else if (c.mem.opcode == VTA_OPCODE_GEMM) {
        // Print instruction field information
        printf("GEMM\n");

        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        printf("\tgemm_opcode: %d\n", static_cast<int>(c.gemm.gemm_opcode));
        printf("\trange (%d, %d)\n", static_cast<int>(c.gemm.uop_bgn),
               static_cast<int>(c.gemm.uop_end));
        printf("\touter loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
               static_cast<int>(c.gemm.iter_out), static_cast<int>(c.gemm.wgt_factor_out),
               static_cast<int>(c.gemm.src_factor_out), static_cast<int>(c.gemm.dst_factor_out));
        printf("\tinner loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
               static_cast<int>(c.gemm.iter_in), static_cast<int>(c.gemm.wgt_factor_in),
               static_cast<int>(c.gemm.src_factor_in), static_cast<int>(c.gemm.dst_factor_in));
      } else if (c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        printf("ALU - %s\n", getOpcodeString(c.alu.alu_opcode, c.alu.use_imm));
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        printf("\treset_out: %d\n", static_cast<int>(c.alu.reset_reg));
        printf("\trange (%d, %d)\n", static_cast<int>(c.alu.uop_bgn),
               static_cast<int>(c.alu.uop_end));
        printf("\touter loop - iter: %d, dst: %d, src: %d\n", static_cast<int>(c.alu.iter_out),
               static_cast<int>(c.alu.dst_factor_out), static_cast<int>(c.alu.src_factor_out));
        printf("\tinner loop - iter: %d, dst: %d, src: %d\n", static_cast<int>(c.alu.iter_in),
               static_cast<int>(c.alu.dst_factor_in), static_cast<int>(c.alu.src_factor_in));
      } else if (c.mem.opcode == VTA_OPCODE_FINISH) {
        printf("FINISH\n");
      }

      // Count status in queues
      if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.opcode == VTA_OPCODE_STORE) {
          CHECK(c.mem.pop_next_dep == false);
          CHECK(c.mem.push_next_dep == false);
          if (c.mem.pop_prev_dep) g2s_queue--;
          if (c.mem.push_prev_dep) s2g_queue++;
        } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                   (c.mem.memory_type == VTA_MEM_ID_INP || c.mem.memory_type == VTA_MEM_ID_WGT)) {
          CHECK(c.mem.pop_prev_dep == false);
          CHECK(c.mem.push_prev_dep == false);
          if (c.mem.pop_next_dep) g2l_queue--;
          if (c.mem.push_next_dep) l2g_queue++;
        } else {
          if (c.mem.pop_prev_dep) l2g_queue--;
          if (c.mem.push_prev_dep) g2l_queue++;
          if (c.mem.pop_next_dep) s2g_queue--;
          if (c.mem.push_next_dep) g2s_queue++;
        }
      } else if (c.mem.opcode == VTA_OPCODE_GEMM || c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        if (c.gemm.pop_prev_dep) l2g_queue--;
        if (c.gemm.push_prev_dep) g2l_queue++;
        if (c.gemm.pop_next_dep) s2g_queue--;
        if (c.gemm.push_next_dep) g2s_queue++;
      }
      printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
      printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
    }
  }
  // Commit all pending pop of corresponding stage
  void CommitPendingPop(int stage) {
    // Handle the LD<->compute queue
    // NOTE: pop executes on target(stage)
    CHECK(stage > 0 && stage < 4);
    if (pending_pop_prev_[stage] || pending_pop_next_[stage]) {
      PushNoop(stage, false, false, pending_pop_prev_[stage], pending_pop_next_[stage]);
      pending_pop_prev_[stage] = 0;
      pending_pop_next_[stage] = 0;
    }
  }
  void CommitPending() {
    for (int i = kLoadStage; i <= kStoreStage; ++i) {
      CommitPendingPop(i);
    }
  }
  bool PendingPop() {
    for (int i = kLoadStage; i <= kStoreStage; ++i) {
      if (pending_pop_prev_[i]) return true;
      if (pending_pop_next_[i]) return true;
    }
    return false;
  }
  void AutoReadBarrier() { ReadBarrier(); }
  /*! \brief Writer barrier to make sure that data written by CPU is visible to VTA. */
  void ReadBarrier() {
    CHECK(fpga_buff_ != nullptr);
    CHECK(fpga_buff_phy_);
    uint32_t buff_size = dram_buffer_.size() * elem_bytes_;
    CHECK(buff_size <= kMaxBytes);
    // Copy contents of DRAM buffer to FPGA buff
    VTAMemCopyFromHost(fpga_buff_, dram_buffer_.data(), buff_size);
    // Flush if we're using a shared memory system
    // and if interface is non-coherent
    if (!coherent_ && always_cache_) {
      VTAFlushCache(fpga_buff_, fpga_buff_phy_, buff_size);
    }
  }

 protected:
  /*! \return Add new instruction to the buffer. */
  VTAGenericInsn* NextInsn() {
    VTAGenericInsn insn;
    memset(&insn, 0, sizeof(insn));
    dram_buffer_.push_back(insn);
    return &dram_buffer_.back();
  }
  // Create a new instruction for a given stage
  VTAGenericInsn* Create(PipelineStage stage) {
    VTAGenericInsn* gptr = NextInsn();
    VTAMemInsn* mptr = reinterpret_cast<VTAMemInsn*>(gptr);
    mptr->pop_prev_dep = pending_pop_prev_[stage];
    mptr->pop_next_dep = pending_pop_next_[stage];
    mptr->push_prev_dep = false;
    mptr->push_next_dep = false;
    pending_pop_prev_[stage] = 0;
    pending_pop_next_[stage] = 0;
    return gptr;
  }
  // Get stage of the memory
  static PipelineStage GetMemPipelineStage(int memory_type) {
    if (memory_type == VTA_MEM_ID_ACC) return kComputeStage;
    if (memory_type == VTA_MEM_ID_UOP) return kComputeStage;
    return kLoadStage;
  }
  // Get stage of the computation
  static PipelineStage GetPipelineStage(VTAMemInsn* insn) {
    if (insn->opcode == VTA_OPCODE_GEMM) return kComputeStage;
    if (insn->opcode == VTA_OPCODE_ALU) return kComputeStage;
    if (insn->opcode == VTA_OPCODE_LOAD) {
      if (insn->x_size == 0) return kNoneStage;
      if (insn->memory_type == VTA_MEM_ID_ACC) return kComputeStage;
      if (insn->memory_type == VTA_MEM_ID_UOP) return kComputeStage;
      return kLoadStage;
    }
    if (insn->opcode == VTA_OPCODE_STORE) {
      // FIXME: Right now memory_type is a 2-bit field which means that
      //        VTA_MEM_ID_OUT will appear as 0. For now we'll refrain from
      //        checking the memory_type to avoid an CHECK error...
      return kStoreStage;
    }
    LOG(FATAL) << "not reached";
    return kNoneStage;
  }

  // Get stage of memory and computation
  static PipelineStage GetPipelineStageAll(VTAMemInsn* insn) {
    PipelineStage stage = GetPipelineStage(insn);
    if (stage != kNoneStage) return stage;
    return GetMemPipelineStage(insn->memory_type);
  }

  // Push no-op
  void PushNoop(int stage, bool push_prev_dep, bool push_next_dep, bool pop_prev_dep,
                bool pop_next_dep) {
    VTAMemInsn* insn = reinterpret_cast<VTAMemInsn*>(NextInsn());
    insn->opcode = (stage == kStoreStage ? VTA_OPCODE_STORE : VTA_OPCODE_LOAD);
    insn->push_prev_dep = push_prev_dep;
    insn->push_next_dep = push_next_dep;
    insn->pop_prev_dep = pop_prev_dep;
    insn->pop_next_dep = pop_next_dep;
    insn->sram_base = 0;
    insn->dram_base = 0;
    insn->y_size = 0;
    insn->x_size = 0;
    insn->x_stride = 0;
    insn->y_pad_0 = 0;
    insn->y_pad_1 = 0;
    insn->x_pad_0 = 0;
    insn->x_pad_1 = 0;
    insn->memory_type = (stage == kLoadStage ? VTA_MEM_ID_INP : VTA_MEM_ID_UOP);
  }

 private:
  // Pending pop of each isntruction queue, qid=0 is not used
  int pending_pop_prev_[4];
  int pending_pop_next_[4];
  static constexpr int kElemBytes = sizeof(VTAGenericInsn);
  static constexpr int kMaxElems = kMaxBytes / kElemBytes;
};

/*!
 * \brief The command queue object that handles the request.
 */
class CommandQueue {
 public:
  CommandQueue() { this->InitSpace(); }
  void InitSpace() {
    uop_queue_.InitSpace();
    insn_queue_.InitSpace();
    device_ = VTADeviceAlloc();
    CHECK(device_ != nullptr);
  }

  ~CommandQueue() {
    VTADeviceFree(device_);
    insn_queue_.FreeSpace();
    uop_queue_.FreeSpace();
    DataBuffer::Collect();
  }

  uint32_t GetElemBytes(uint32_t memory_id) {
    uint32_t elem_bytes = 0;
    switch (memory_id) {
      case VTA_MEM_ID_UOP:
        elem_bytes = VTA_UOP_ELEM_BYTES;
        break;
      case VTA_MEM_ID_INP:
        elem_bytes = VTA_INP_ELEM_BYTES;
        break;
      case VTA_MEM_ID_WGT:
        elem_bytes = VTA_WGT_ELEM_BYTES;
        break;
      case VTA_MEM_ID_ACC:
        elem_bytes = VTA_ACC_ELEM_BYTES;
        break;
      case VTA_MEM_ID_OUT:
        elem_bytes = VTA_OUT_ELEM_BYTES;
        break;
      default:
        LOG(FATAL) << "Memory id not recognized:" << memory_id;
        break;
    }
    /*
     * elements size should not larger than VTA_PAGE_BYTES.
     *
     */
    CHECK_GE(VTA_PAGE_BYTES, elem_bytes);
    return elem_bytes;
  }

  void LoadBuffer2D(void* src_dram_addr, uint32_t src_elem_offset, uint32_t x_size, uint32_t y_size,
                    uint32_t x_stride, uint32_t x_pad_before, uint32_t y_pad_before,
                    uint32_t x_pad_after, uint32_t y_pad_after, uint32_t is_pad_min_value, uint32_t dst_sram_index,
                    uint32_t dst_memory_type) {
    VTAMemInsn* insn = insn_queue_.CreateMemInsn(dst_memory_type);
    insn->opcode = VTA_OPCODE_LOAD;
    insn->memory_type = dst_memory_type;
    insn->sram_base = dst_sram_index;
    DataBuffer* src = DataBuffer::FromHandle(src_dram_addr);
    insn->dram_base = src->phy_addr() / GetElemBytes(dst_memory_type) + src_elem_offset;
    insn->y_size = y_size;
    insn->x_size = x_size;
    insn->x_stride = x_stride;
    insn->y_pad_0 = y_pad_before;
    insn->y_pad_1 = y_pad_after;
    insn->x_pad_0 = x_pad_before;
    insn->x_pad_1 = x_pad_after;
    insn->is_pad_min_value = is_pad_min_value;
    this->CheckInsnOverFlow();
    if (trace_mgr.TraceLoad()) {
      // Print out the memory region that will be read by the load instruction
      // during the run for the layer. At this point the memory has already been
      // copied from the host so we can read the dram and dump for x-reference.
      uint32_t elem_bytes = GetElemBytes(dst_memory_type);
      // PC: There is a bug in the encoding of the memory type in the
      // memory instruction. Field memory_type has two bits, but there are 5
      // different memory types that are assigned to it. VTA_MEM_ID_OUT == 4
      // will be truncated to VTA_MEM_ID_UOP == 0. This means that whenever
      // VTA_OUT_ELEM_BYTES != VTA_UOP_ELEM_BYTES hell breaks loose.
      // if (insn->memory_type != dst_memory_type)
      //{
      //  printf ("MEMORY TYPE mismatch: 'encoded %u != expected %u'\n",
      //          (unsigned)insn->memory_type, (unsigned)dst_memory_type);
      //}
      // assert (insn->memory_type == dst_memory_type);
      assert(src->phy_addr() % elem_bytes == 0);
      uint32_t size_bytes = x_size * y_size * elem_bytes;
      void* virt_addr = (uint8_t*)src->virt_addr() + src_elem_offset * elem_bytes;
      uint32_t phys_addr = src->phy_addr() + src_elem_offset * elem_bytes;
      uint32_t sram_addr = insn->sram_base * elem_bytes;  // can this be truncated?
      const char* mem_type = "NOP";
      if (insn->memory_type == VTA_MEM_ID_UOP)
        mem_type = "LD_UOP";
      else if (insn->memory_type == VTA_MEM_ID_WGT)
        mem_type = "LD_WGT";
      else if (insn->memory_type == VTA_MEM_ID_INP)
        mem_type = "LD_INP";
      else if (insn->memory_type == VTA_MEM_ID_ACC)
        mem_type = "LD_ACC";
      void* vap = trace_mgr.TraceVirtAddr() ? virt_addr : NULL;
      printf(mem_fmt, mem_type, vap, phys_addr, sram_addr, elem_bytes, size_bytes);
      if (trace_mgr.TraceXY()) {
        printf("y_size: %" PRIu64 "\n", insn->y_size);
        printf("x_size: %" PRIu64 "\n", insn->x_size);
        printf("x_strd: %" PRIu64 "\n", insn->x_stride);
        printf("y_pad0: %" PRIu64 "\n", insn->y_pad_0);
        printf("y_pad1: %" PRIu64 "\n", insn->y_pad_1);
        printf("x_pad0: %" PRIu64 "\n", insn->x_pad_0);
        printf("x_pad1: %" PRIu64 "\n", insn->x_pad_1);
        printf("is_pad_min_value: %" PRIu64 "\n", insn->is_pad_min_value);
      }
      // The sram addr in bytes, not elements.
      uint8_t *next_addr = (uint8_t*)virt_addr, *dram_addr;
      // The total load into sram includes padding at beginning and end.
      uint64_t xtotal = insn->x_size + insn->x_pad_0 + insn->x_pad_1;
      // uint32_t ytotal = insn->y_size + insn->y_pad_0 + insn->y_pad_1;
      // Advance sram by head padding.
      sram_addr += xtotal * insn->y_pad_0 * elem_bytes;
      uint32_t xfer_bytes = 16 * 8, phys_next = phys_addr;

      if (trace_mgr.TraceVirtAddr()) printf(" %10s", "virt_addr");
      printf(" %10s %10s", "phys_addr", "sram_addr");
      if (trace_mgr.TraceXY()) printf(" %4s %4s", "y", "x");
      printf(" dma transfer\n");
      for (uint32_t y = 0; y < insn->y_size; ++y) {
        // memset(sram_ptr, 0, kElemBytes * op->x_pad_0);
        sram_addr += insn->x_pad_0 * elem_bytes;
        dram_addr = next_addr;
        phys_addr = phys_next;
        // Actual load from dram to sram.
        // memcpy(sram_ptr, dram_ptr, kElemBytes * op->x_size);
        // In the VTA implementation, each x read, initial and stride is done with a
        // sequence of DMA transfers. Check that we do not cross a page boundary
        // during a tranfer. Dump memory one transfer per line.
        uint32_t b = 0;
        bool page_cross = false;
        for (uint32_t x = 0; x < insn->x_size; ++x) {
          for (uint32_t z = 0; z < elem_bytes; ++z) {
            if (b % xfer_bytes) {
              // Within a DMA trasfer, have we crossed a page boundary?
              uint32_t this_page = phys_addr & 0xFFFF8000;
              uint32_t prev_page = (phys_addr - 1) & 0xFFFF8000;
              if (this_page != prev_page)
                page_cross = true;
            } else {
              if (trace_mgr.TraceVirtAddr()) printf(" 0x%08" PRIxPTR, dram_addr);
              printf(" 0x%08x 0x%08x", phys_addr, sram_addr);
              if (trace_mgr.TraceXY()) printf(" %4" PRIu32 " %4" PRIu32, y, x);
            }
            printf(" %02x", *dram_addr++);
            phys_addr++;
            sram_addr++;
            if (++b % xfer_bytes == 0) {
              printf("\n");
              if (page_cross) {
                printf("LOAD_PAGE_BOUNDARY_CROSSING\n");
                page_cross = false;
              }
            }
          }
        }
        if (b % xfer_bytes) printf("\n");
        // memset(sram_ptr, 0, kElemBytes * op->x_pad_1);
        sram_addr += insn->x_pad_1 * elem_bytes;
        next_addr += elem_bytes * insn->x_stride;
        phys_next += elem_bytes * insn->x_stride;
      }
      // Tail padding.
      // memset(sram_ptr, 0, kElemBytes * xtotal * op->y_pad_1);
      trace_mgr.Flush();
    }
  }

  std::vector<void*> m_store_virt;

  void StoreBuffer2D(uint32_t src_sram_index, uint32_t src_memory_type, void* dst_dram_addr,
                     uint32_t dst_elem_offset, uint32_t x_size, uint32_t y_size,
                     uint32_t x_stride) {
    VTAMemInsn* insn = insn_queue_.CreateStoreInsn();
    insn->opcode = VTA_OPCODE_STORE;
    insn->memory_type = src_memory_type;
    insn->sram_base = src_sram_index;
    DataBuffer* dst = DataBuffer::FromHandle(dst_dram_addr);
    insn->dram_base = dst->phy_addr() / GetElemBytes(src_memory_type) + dst_elem_offset;
    insn->y_size = y_size;
    insn->x_size = x_size;
    insn->x_stride = x_stride;
    insn->y_pad_0 = 0;
    insn->y_pad_1 = 0;
    insn->x_pad_0 = 0;
    insn->x_pad_1 = 0;
    insn->is_pad_min_value = 0;
    this->CheckInsnOverFlow();
    if (trace_mgr.TraceStore()) {
      uint32_t insn_idx = insn_queue_.count() - 1;
      // Print out the memory region that will be written by the store instruction
      // during the run for the layer. We would need to cache this and dump later
      // after the run has ended for x-reference.
      uint32_t elem_bytes = GetElemBytes(src_memory_type);
      // if (insn->memory_type != src_memory_type)
      //{
      //  printf ("MEMORY TYPE mismatch: 'encoded %u != expected %u'\n",
      //          (unsigned)insn->memory_type, (unsigned)src_memory_type);
      //}
      // assert (insn->memory_type == src_memory_type);
      assert(dst->phy_addr() % elem_bytes == 0);
      assert((uint64_t)dst->virt_addr() % elem_bytes == 0);
      uint32_t size_bytes = x_size * y_size * elem_bytes;
      void* virt_addr = (uint8_t*)dst->virt_addr() + dst_elem_offset * elem_bytes;
      uint32_t phys_addr = dst->phy_addr() + dst_elem_offset * elem_bytes;
      uint32_t sram_addr = src_sram_index * elem_bytes;
      void* vap = trace_mgr.TraceVirtAddr() ? virt_addr : NULL;
      printf(mem_fmt, "ST_OUT", vap, phys_addr, sram_addr, elem_bytes, size_bytes);
      if (trace_mgr.TraceXY()) {
        printf("y_size: %" PRIu64 "\n", insn->y_size);
        printf("x_size: %" PRIu64 "\n", insn->x_size);
        printf("x_strd: %" PRIu64 "\n", insn->x_stride);
        printf("y_pad0: %" PRIu64 "\n", insn->y_pad_0);
        printf("y_pad1: %" PRIu64 "\n", insn->y_pad_1);
        printf("x_pad0: %" PRIu64 "\n", insn->x_pad_0);
        printf("x_pad1: %" PRIu64 "\n", insn->x_pad_1);
        printf("is_pad_min_value: %" PRIu64 "\n", insn->is_pad_min_value);
      }
      if (insn_idx >= m_store_virt.size()) m_store_virt.resize(insn_idx + 1);
      m_store_virt[insn_idx] = virt_addr;
    }
  }

  void DepPush(int from_qid, int to_qid) { insn_queue_.DepPush(from_qid, to_qid); }

  void DepPop(int from_qid, int to_qid) { insn_queue_.DepPop(from_qid, to_qid); }

  void ReadBarrier(void* buffer, uint32_t elem_bits, uint32_t start, uint32_t extent) {
    if (!(debug_flag_ & VTA_DEBUG_SKIP_READ_BARRIER)) {
      uint32_t elem_bytes = (elem_bits + 8 - 1) / 8;
      DataBuffer::FromHandle(buffer)->FlushCache(elem_bytes * start, elem_bytes * extent);
    }
  }

  void WriteBarrier(void* buffer, uint32_t elem_bits, uint32_t start, uint32_t extent) {
    if (!(debug_flag_ & VTA_DEBUG_SKIP_WRITE_BARRIER)) {
      uint32_t elem_bytes = (elem_bits + 8 - 1) / 8;
      DataBuffer::FromHandle(buffer)->InvalidateCache(elem_bytes * start, elem_bytes * extent);
    }
  }

  void Synchronize(uint32_t wait_cycles) {
    // Insert dependences to force serialization
    if (debug_flag_ & VTA_DEBUG_FORCE_SERIAL || trace_mgr.EventEnabled("SERIAL")) {
      insn_queue_.RewriteForceSerial();
    } else {
      // This will issue finish after last store finishes
      insn_queue_.DepPush(kStoreStage, kComputeStage);
      insn_queue_.DepPush(kLoadStage, kComputeStage);
      insn_queue_.DepPop(kStoreStage, kComputeStage);
      insn_queue_.DepPop(kLoadStage, kComputeStage);
      insn_queue_.CommitPendingPop(kComputeStage);
    }
    // NOTE: FINISH cannot contain pop
    VTAGemInsn* insn = insn_queue_.CreateGemInsn();
    insn->opcode = VTA_OPCODE_FINISH;
    CHECK(!insn_queue_.PendingPop());
    // Check if there are no instruction to execute at all
    if (insn_queue_.count() == 0) return;
    // Synchronization for the queues
    uop_queue_.AutoReadBarrier();
    insn_queue_.AutoReadBarrier();
    // Dump instructions if debug enabled
    if (debug_flag_ & VTA_DEBUG_DUMP_INSN) {
      insn_queue_.DumpInsn();
    }
    // Make sure that the last instruction is a finish instruction
    CHECK(reinterpret_cast<VTAMemInsn*>(insn_queue_.data())[insn_queue_.count() - 1].opcode ==
          VTA_OPCODE_FINISH);

    // Make sure that we don't exceed contiguous physical memory limits
    CHECK(insn_queue_.count() * sizeof(VTAGenericInsn) < VTA_MAX_XFER);
    if (trace_mgr.TraceInsn()) {
      printf("VTADeviceRun: insn_count: %d, dram_phy_addr: 0x%x\n", insn_queue_.count(),
             insn_queue_.dram_phy_addr());
      uop_queue_.DumpUop();
      insn_queue_.DumpInsn();
    }
    if (trace_mgr.EventEnabled("ISSUE")) {
      // union VTAInsn c;
      int insn_count = insn_queue_.count();
      const VTAGenericInsn* i = insn_queue_.data();
      for (int x = 0; x < insn_count; ++x, ++i) {
        // c.generic = ginsn[i];
        trace_mgr.Event("ISSUE", "%4u %016" PRIx64 "%016" PRIx64 "\n",
        x, *((uint64_t*)i + 1), *((uint64_t*)i));
      }
    }
    int timeout =
        VTADeviceRun(device_, insn_queue_.dram_phy_addr(), insn_queue_.count(), wait_cycles);
    CHECK_EQ(timeout, 0);
    if (trace_mgr.TraceStore()) {
      union VTAInsn c;
      int insn_count = insn_queue_.count();
      const VTAGenericInsn* ginsn = insn_queue_.data();
      for (int i = 0; i < insn_count; ++i) {
        c.generic = ginsn[i];
        if (c.mem.opcode == VTA_OPCODE_LOAD) {
          if (c.mem.x_size == 0) continue;
        } else if (c.mem.opcode == VTA_OPCODE_STORE) {
          if (c.mem.x_size == 0) continue;
          // BUG here: c.mem.memory_type is truncated to 0 from 4.
          // uint32_t elem_bytes = GetElemBytes(c.mem.memory_type);
          uint32_t elem_bytes = 16, xfer_bytes = 16 * 8; // de10nano
          uint32_t size_bytes = c.mem.x_size * c.mem.y_size * elem_bytes;
          uint8_t* virt_addr = (uint8_t*)m_store_virt[i];
          uint8_t* phys_addr = (uint8_t*)(c.mem.dram_base * elem_bytes);
          uint32_t sram_addr = c.mem.sram_base * elem_bytes, s;
          void* vap = trace_mgr.TraceVirtAddr() ? virt_addr : NULL;
          printf(mem_fmt, "ST_OUT", vap, phys_addr, sram_addr, elem_bytes, size_bytes);
          uint8_t *v, *p;
          bool page_cross = false;
          if (trace_mgr.TraceVirtAddr()) printf(" %10s", "virt_addr");
          printf(" %10s %10s", "phys_addr", "sram_addr");
          if (trace_mgr.TraceXY()) printf(" %4s %4s", "y", "x");
          printf(" elements\n");
          for (uint32_t y = 0; y < c.mem.y_size; ++y) {
            uint32_t b = 0; // start of dma transfer
            for (uint32_t x = 0; x < c.mem.x_size; ++x) {
              v = virt_addr + (y * c.mem.x_stride + x) * elem_bytes;
              p = phys_addr + (y * c.mem.x_stride + x) * elem_bytes;
              s = sram_addr + (y * c.mem.x_size + x) * elem_bytes;
              if (trace_mgr.TraceVirtAddr()) printf(" 0x%08" PRIxPTR, v);
              printf(" 0x%08x 0x%08x", p, s);
              if (trace_mgr.TraceXY()) printf(" %4" PRIu32 " %4" PRIu32, y, x);
              for (uint32_t j = 0; j < elem_bytes; ++j, b++, p++) {
                if (b % xfer_bytes) {
                  // Within a DMA trasfer, have we crossed a page boundary?
                  uint32_t prev_page = ((uint64_t)(p-1)) & 0xFFFF8000;
                  uint32_t this_page = ((uint64_t)p) & 0xFFFF8000;
                  if (this_page != prev_page) {
                    printf("!%02x", *v++);
                    page_cross = true;
                  } else
                    printf(" %02x", *v++);
                } else
                  printf("+%02x", *v++);
              }
              printf("\n");
              if (page_cross) {
                printf("STORE_PAGE_BOUNDARY_CROSSING\n");
                page_cross = false;
              }
            }
          }
        }
      }
    }
    // Reset buffers
    uop_queue_.Reset();
    insn_queue_.Reset();
  }

  // Get record kernel
  UopKernel* record_kernel() const {
    CHECK(record_kernel_ != nullptr);
    return record_kernel_;
  }

  // Set debug flag
  void SetDebugFlag(int debug_flag) { debug_flag_ = debug_flag; }

  void PushGEMMOp(void** uop_handle, int (*finit)(void*), void* signature, int nbytes) {
    UopKernelMap** uptr = reinterpret_cast<UopKernelMap**>(uop_handle);
    if (uptr[0] == nullptr) {
      uptr[0] = new UopKernelMap();
    }
    UopKernel** kptr = uptr[0]->Get(signature, nbytes);
    if (kptr[0] == nullptr) {
      record_kernel_ = new UopKernel(static_cast<char*>(signature), nbytes);
      CHECK_EQ((*finit)(signature), 0);
      kptr[0] = static_cast<UopKernel*>(record_kernel_);
      if (debug_flag_ & VTA_DEBUG_DUMP_UOP) {
        record_kernel_->Dump();
      }
      record_kernel_ = nullptr;
    }
    this->PushGEMMOp(static_cast<UopKernel*>(kptr[0]));
    this->CheckInsnOverFlow();
  }

  void PushALUUop(void** uop_handle, int (*finit)(void*), void* signature, int nbytes) {
    UopKernelMap** uptr = reinterpret_cast<UopKernelMap**>(uop_handle);
    if (uptr[0] == nullptr) {
      uptr[0] = new UopKernelMap();
    }
    UopKernel** kptr = uptr[0]->Get(signature, nbytes);
    if (kptr[0] == nullptr) {
      record_kernel_ = new UopKernel(static_cast<char*>(signature), nbytes);
      CHECK_EQ((*finit)(signature), 0);
      kptr[0] = static_cast<UopKernel*>(record_kernel_);
      if (debug_flag_ & VTA_DEBUG_DUMP_UOP) {
        record_kernel_->Dump();
      }
      record_kernel_ = nullptr;
    }
    this->PushALUUop(static_cast<UopKernel*>(kptr[0]));
    this->CheckInsnOverFlow();
  }

  static std::shared_ptr<CommandQueue>& ThreadLocal() {
    static std::shared_ptr<CommandQueue> inst = std::make_shared<CommandQueue>();
    if (inst == nullptr) {
      inst = std::make_shared<CommandQueue>();
    }
    return inst;
  }

  static void Shutdown() { ThreadLocal().reset(); }

 private:
  // Push GEMM uop to the command buffer
  void PushGEMMOp(UopKernel* kernel) {
    uop_queue_.Push(kernel, [this]() { this->AutoSync(); });
    if (uop_queue_.pending()) {
      uop_queue_.FlushUopLoad(insn_queue_);
    }
    VTAGemInsn* insn = insn_queue_.CreateGemInsn();
    insn->opcode = VTA_OPCODE_GEMM;
    // if reset then opcode 0 else 1 (or 2 or 3 for depthwise)
    switch (kernel->reset_out_) {
      case 0:
        insn->gemm_opcode = VTA_GEMM_OPCODE_NORMAL;
      break;
      case 1:
        insn->gemm_opcode = VTA_GEMM_OPCODE_RESET;
      break;
      case 2:
        insn->gemm_opcode = VTA_GEMM_OPCODE_DEPTHWISE_STRIDE1;
      break;
      case 3:
        insn->gemm_opcode = VTA_GEMM_OPCODE_DEPTHWISE_STRIDE2;
    }
    insn->uop_bgn = kernel->sram_begin_;
    insn->uop_end = kernel->sram_end_;
    const std::vector<UopKernel::LoopEntry>& loop = kernel->loop();
    if (loop.size() > 0) {
      assert(loop[0].extent < (1 << VTA_LOOP_ITER_WIDTH));
      assert(loop[0].wgt_factor < (1 << VTA_WGT_FACTOR_WIDTH));
      assert(loop[0].src_factor < (1 << VTA_INP_FACTOR_WIDTH));
      assert(loop[0].dst_factor < (1 << VTA_ACC_FACTOR_WIDTH));
      insn->iter_out = loop[0].extent;
      insn->wgt_factor_out = loop[0].wgt_factor;
      insn->src_factor_out = loop[0].src_factor;
      insn->dst_factor_out = loop[0].dst_factor;
    } else {
      insn->iter_out = 1;
      insn->wgt_factor_out = 0;
      insn->src_factor_out = 0;
      insn->dst_factor_out = 0;
    }
    if (loop.size() > 1) {
      assert(loop[1].extent < (1 << VTA_LOOP_ITER_WIDTH));
      assert(loop[1].wgt_factor < (1 << VTA_WGT_FACTOR_WIDTH));
      assert(loop[1].src_factor < (1 << VTA_INP_FACTOR_WIDTH));
      assert(loop[1].dst_factor < (1 << VTA_ACC_FACTOR_WIDTH));    
      insn->iter_in = loop[1].extent;
      insn->wgt_factor_in = loop[1].wgt_factor;
      insn->src_factor_in = loop[1].src_factor;
      insn->dst_factor_in = loop[1].dst_factor;
    } else {
      insn->iter_in = 1;
      insn->wgt_factor_in = 0;
      insn->src_factor_in = 0;
      insn->dst_factor_in = 0;
    }
  }

  // Push ALU uop to the command buffer
  void PushALUUop(UopKernel* kernel) {
    uop_queue_.Push(kernel, [this]() { this->AutoSync(); });
    if (uop_queue_.pending()) {
      uop_queue_.FlushUopLoad(insn_queue_);
    }
    VTAAluInsn* insn = insn_queue_.CreateAluInsn();
    insn->opcode = VTA_OPCODE_ALU;
    insn->reset_reg = kernel->reset_out_;
    insn->uop_bgn = kernel->sram_begin_;
    insn->uop_end = kernel->sram_end_;
    assert(kernel->opcode_ < (1 << VTA_ALU_OPCODE_BIT_WIDTH));
    insn->alu_opcode = kernel->opcode_;
    insn->use_imm = kernel->use_imm_;
    insn->imm = kernel->imm_val_;
    const std::vector<UopKernel::LoopEntry>& loop = kernel->loop();
    if (loop.size() == 0) {
      insn->iter_out = 1;
      insn->dst_factor_out = 0;
      insn->src_factor_out = 0;
      insn->iter_in = 1;
      insn->dst_factor_in = 0;
      insn->src_factor_in = 0;
    } else if (loop.size() == 1) {
      insn->iter_out = 1;
      insn->dst_factor_out = 0;
      insn->src_factor_out = 0;
      assert(loop[0].extent < (1 << VTA_LOOP_ITER_WIDTH));
      assert(loop[0].dst_factor < (1 << VTA_ACC_FACTOR_WIDTH));
      assert(loop[0].src_factor < (1 << VTA_INP_FACTOR_WIDTH));
      insn->iter_in = loop[0].extent;
      insn->dst_factor_in = loop[0].dst_factor;
      insn->src_factor_in = loop[0].src_factor;
    } else {
      insn->iter_out = loop[0].extent;
      insn->dst_factor_out = loop[0].dst_factor;
      insn->src_factor_out = loop[0].src_factor;
      assert(loop[1].extent < (1 << VTA_LOOP_ITER_WIDTH));
      assert(loop[1].dst_factor < (1 << VTA_ACC_FACTOR_WIDTH));
      assert(loop[1].src_factor < (1 << VTA_INP_FACTOR_WIDTH));      
      insn->iter_in = loop[1].extent;
      insn->dst_factor_in = loop[1].dst_factor;
      insn->src_factor_in = loop[1].src_factor;
    }
  }

  void CheckInsnOverFlow() {
    // At each API call, we can at most commit:
    // one pending store, one pending load, and one uop
    if ((insn_queue_.count() + 4) * sizeof(VTAGenericInsn) >= VTA_MAX_XFER) {
      this->AutoSync();
    }
  }
  // Auto sync when instruction overflow
  void AutoSync() { this->Synchronize(1 << 31); }

  // Internal debug flag
  int debug_flag_{0};
  // The kernel we are currently recording
  UopKernel* record_kernel_{nullptr};
  // Micro op queue
  UopQueue<VTA_MAX_XFER, kBufferCoherent, kAlwaysCache> uop_queue_;
  // instruction queue
  InsnQueue<VTA_MAX_XFER, kBufferCoherent, kAlwaysCache> insn_queue_;
  // Device handle
  VTADeviceHandle device_{nullptr};
};

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;

TVM_REGISTER_GLOBAL("vta.runtime.trace.enable").set_body([](TVMArgs args, TVMRetValue* rv) {
  trace_mgr.Trace(args[0]);
  *rv = trace_mgr.Trace();
});

TVM_REGISTER_GLOBAL("vta.runtime.trace.enable_event").set_body([](TVMArgs args, TVMRetValue* rv) {
  trace_mgr.EnableEvent(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("vta.runtime.trace.flush").set_body([](TVMArgs args, TVMRetValue* rv) {
  trace_mgr.Flush();
});

TVM_REGISTER_GLOBAL("vta.runtime.trace.redirect_stdout")
    .set_body([](TVMArgs args, TVMRetValue* rv) { trace_mgr.RedirectStdout(); });

TVM_REGISTER_GLOBAL("vta.runtime.trace.quit").set_body([](TVMArgs args, TVMRetValue* rv) {
  trace_mgr.Quit();
});

TVM_REGISTER_GLOBAL("vta.runtime.trace.init").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = trace_mgr.Init(args[0]);
});

}  // namespace vta

void* VTABufferAlloc(size_t size) { return vta::DataBuffer::Alloc(size); }

void VTABufferFree(void* buffer) { vta::DataBuffer::Free(vta::DataBuffer::FromHandle(buffer)); }

void VTABufferCopy(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                   int kind_mask) {
  vta::DataBuffer* from_buffer = nullptr;
  vta::DataBuffer* to_buffer = nullptr;

  if (kind_mask & 2) {
    from_buffer = vta::DataBuffer::FromHandle(from);
    from = from_buffer->virt_addr();
  }
  if (kind_mask & 1) {
    to_buffer = vta::DataBuffer::FromHandle(to);
    to = to_buffer->virt_addr();
  }

  if (from_buffer) {
    // This is an FPGA to host mem transfer
    from_buffer->InvalidateCache(from_offset, size);
    from_buffer->MemCopyToHost(static_cast<char*>(to) + to_offset,
                               static_cast<const char*>(from) + from_offset, size);
  } else if (to_buffer) {
    // This is a host to FPGA mem transfer
    to_buffer->MemCopyFromHost(static_cast<char*>(to) + to_offset,
                               static_cast<const char*>(from) + from_offset, size);
    to_buffer->FlushCache(to_offset, size);
  }
}

VTACommandHandle VTATLSCommandHandle() { return vta::CommandQueue::ThreadLocal().get(); }

void VTARuntimeShutdown() { vta::CommandQueue::Shutdown(); }

void VTASetDebugMode(VTACommandHandle cmd, int debug_flag) {
  static_cast<vta::CommandQueue*>(cmd)->SetDebugFlag(debug_flag);
}

void* VTABufferCPUPtr(VTACommandHandle cmd, void* buffer) {
  return vta::DataBuffer::FromHandle(buffer)->virt_addr();
}

void VTAWriteBarrier(VTACommandHandle cmd, void* buffer, uint32_t elem_bits, uint32_t start,
                     uint32_t extent) {
  static_cast<vta::CommandQueue*>(cmd)->WriteBarrier(buffer, elem_bits, start, extent);
}

void VTAReadBarrier(VTACommandHandle cmd, void* buffer, uint32_t elem_bits, uint32_t start,
                    uint32_t extent) {
  static_cast<vta::CommandQueue*>(cmd)->ReadBarrier(buffer, elem_bits, start, extent);
}

void VTALoadBuffer2D(VTACommandHandle cmd, void* src_dram_addr, uint32_t src_elem_offset,
                     uint32_t x_size, uint32_t y_size, uint32_t x_stride, uint32_t x_pad_before,
                     uint32_t y_pad_before, uint32_t x_pad_after, uint32_t y_pad_after,
                     uint32_t is_pad_min_value, uint32_t dst_sram_index, uint32_t dst_memory_type) {
  static_cast<vta::CommandQueue*>(cmd)->LoadBuffer2D(
      src_dram_addr, src_elem_offset, x_size, y_size, x_stride, x_pad_before, y_pad_before,
      x_pad_after, y_pad_after, is_pad_min_value, dst_sram_index, dst_memory_type);
}

void VTAStoreBuffer2D(VTACommandHandle cmd, uint32_t src_sram_index, uint32_t src_memory_type,
                      void* dst_dram_addr, uint32_t dst_elem_offset, uint32_t x_size,
                      uint32_t y_size, uint32_t x_stride) {
  static_cast<vta::CommandQueue*>(cmd)->StoreBuffer2D(
      src_sram_index, src_memory_type, dst_dram_addr, dst_elem_offset, x_size, y_size, x_stride);
}

void VTAUopPush(uint32_t mode, uint32_t reset_out, uint32_t dst_index, uint32_t src_index,
                uint32_t wgt_index, uint32_t opcode, uint32_t use_imm, int32_t imm_val) {
  vta::CommandQueue::ThreadLocal()->record_kernel()->Push(mode, reset_out, dst_index, src_index,
                                                          wgt_index, opcode, use_imm, imm_val);
}

void VTAUopLoopBegin(uint32_t extent, uint32_t dst_factor, uint32_t src_factor,
                     uint32_t wgt_factor) {
  vta::CommandQueue::ThreadLocal()->record_kernel()->PushLoopBegin(extent, dst_factor, src_factor,
                                                                   wgt_factor);
}

void VTAUopLoopEnd() { vta::CommandQueue::ThreadLocal()->record_kernel()->PushLoopEnd(); }

int VTAPushGEMMOp(void** uop_handle, int (*finit)(void*), void* signature, int nbytes) {
  vta::CommandQueue::ThreadLocal()->PushGEMMOp(uop_handle, finit, signature, nbytes);
  return 0;
}

int VTAPushALUOp(void** uop_handle, int (*finit)(void*), void* signature, int nbytes) {
  vta::CommandQueue::ThreadLocal()->PushALUUop(uop_handle, finit, signature, nbytes);
  return 0;
}

int VTADepPush(VTACommandHandle cmd, int from_qid, int to_qid) {
  static_cast<vta::CommandQueue*>(cmd)->DepPush(from_qid, to_qid);
  return 0;
}

int VTADepPop(VTACommandHandle cmd, int from_qid, int to_qid) {
  static_cast<vta::CommandQueue*>(cmd)->DepPop(from_qid, to_qid);
  return 0;
}

void VTASynchronize(VTACommandHandle cmd, uint32_t wait_cycles) {
  static_cast<vta::CommandQueue*>(cmd)->Synchronize(wait_cycles);
}
