#include "memory_integrity_tree.h"

MemoryIntegrityTree::MemoryIntegrityTree(size_t data_size) {
  size_t num_leaves = (data_size + kBlockSize - 1) / kBlockSize;
  size_t tree_size = 2 * num_leaves - 1;
  tree_.resize(tree_size);
}

void MemoryIntegrityTree::UpdateHash(size_t index, const void* data, size_t size) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256((unsigned char*)data, size, hash);
  size_t leaf_index = index / kBlockSize + tree_.size() / 2;
  tree_[leaf_index] = std::vector<unsigned char>(hash, hash + SHA256_DIGEST_LENGTH);
  UpdateParentHashes(leaf_index);
}

bool MemoryIntegrityTree::VerifyHash(size_t index, const void* data, size_t size) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256((unsigned char*)data, size, hash);
  size_t leaf_index = index / kBlockSize + tree_.size() / 2;
  if (tree_[leaf_index] != std::vector<unsigned char>(hash, hash + SHA256_DIGEST_LENGTH)) {
    return false;
  }
  return VerifyParentHashes(leaf_index);
}

void MemoryIntegrityTree::UpdateParentHashes(size_t index) {
  while (index > 0) {
    size_t parent = (index - 1) / 2;
    size_t sibling = index % 2 == 0 ? index - 1 : index + 1;
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, tree_[index].data(), tree_[index].size());
    if (sibling < tree_.size()) {
      SHA256_Update(&sha256, tree_[sibling].data(), tree_[sibling].size());
    }
    SHA256_Final(hash, &sha256);
    tree_[parent] = std::vector<unsigned char>(hash, hash + SHA256_DIGEST_LENGTH);
    index = parent;
  }
}

bool MemoryIntegrityTree::VerifyParentHashes(size_t index) {
  while (index > 0) {
    size_t parent = (index - 1) / 2;
    size_t sibling = index % 2 == 0 ? index - 1 : index + 1;
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, tree_[index].data(), tree_[index].size());
    if (sibling < tree_.size()) {
      SHA256_Update(&sha256, tree_[sibling].data(), tree_[sibling].size());
    }
    SHA256_Final(hash, &sha256);
    if (tree_[parent] != std::vector<unsigned char>(hash, hash + SHA256_DIGEST_LENGTH)) {
      return false;
    }
    index = parent;
  }
  return true;
}