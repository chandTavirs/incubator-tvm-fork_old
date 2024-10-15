#ifndef TVM_VTA_RUNTIME_MEMORY_INTEGRITY_TREE_H_
#define TVM_VTA_RUNTIME_MEMORY_INTEGRITY_TREE_H_

#include <vector>
#include <openssl/sha.h>

class MemoryIntegrityTree {
 public:
  explicit MemoryIntegrityTree(size_t data_size);

  void UpdateHash(size_t index, const void* data, size_t size);
  bool VerifyHash(size_t index, const void* data, size_t size);

 private:
  static const size_t kBlockSize = 64;
  std::vector<std::vector<unsigned char>> tree_;

  void UpdateParentHashes(size_t index);
  bool VerifyParentHashes(size_t index);
};

#endif  // TVM_VTA_RUNTIME_MEMORY_INTEGRITY_TREE_H_