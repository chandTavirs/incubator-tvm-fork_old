#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>  // For hashing

#define BLOCK_SIZE 64  // Size of each memory block in bytes
#define HASH_SIZE SHA256_DIGEST_LENGTH

// Node in the Memory Integrity Tree
typedef struct MITNode {
    unsigned char hash[HASH_SIZE];  // Hash of the data or children nodes
    struct MITNode *left;           // Left child
    struct MITNode *right;          // Right child
} MITNode;

// Function to create a new node
MITNode* create_node() {
    MITNode *node = (MITNode *)malloc(sizeof(MITNode));
    memset(node->hash, 0, HASH_SIZE);
    node->left = NULL;
    node->right = NULL;
    return node;
}

// Function to compute the SHA-256 hash of data
void compute_hash(const unsigned char *data, size_t len, unsigned char *hash_out) {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data, len);
    SHA256_Final(hash_out, &sha256);
}

// Function to build the MIT recursively
MITNode* build_mit(unsigned char memory[][BLOCK_SIZE], int start, int end) {
    if (start == end) {
        // Leaf node: hash the memory block
        MITNode *leaf = create_node();
        compute_hash(memory[start], BLOCK_SIZE, leaf->hash);
        return leaf;
    }

    int mid = (start + end) / 2;
    MITNode *node = create_node();
    node->left = build_mit(memory, start, mid);
    node->right = build_mit(memory, mid + 1, end);

    // Internal node: hash the concatenation of the children's hashes
    unsigned char combined_hash[2 * HASH_SIZE];
    memcpy(combined_hash, node->left->hash, HASH_SIZE);
    memcpy(combined_hash + HASH_SIZE, node->right->hash, HASH_SIZE);
    compute_hash(combined_hash, 2 * HASH_SIZE, node->hash);

    return node;
}

// Function to update the MIT after modifying a memory block
void update_mit(MITNode *node, unsigned char memory[][BLOCK_SIZE], int index, int start, int end) {
    if (start == end) {
        // Update the hash of the modified memory block
        compute_hash(memory[start], BLOCK_SIZE, node->hash);
        return;
    }

    int mid = (start + end) / 2;
    if (index <= mid) {
        update_mit(node->left, memory, index, start, mid);
    } else {
        update_mit(node->right, memory, index, mid + 1, end);
    }

    // Recalculate the hash for the current node
    unsigned char combined_hash[2 * HASH_SIZE];
    memcpy(combined_hash, node->left->hash, HASH_SIZE);
    memcpy(combined_hash + HASH_SIZE, node->right->hash, HASH_SIZE);
    compute_hash(combined_hash, 2 * HASH_SIZE, node->hash);
}

// Function to verify the integrity of a memory block
int verify_mit(MITNode *node, unsigned char memory[][BLOCK_SIZE], int index, int start, int end) {
    if (start == end) {
        unsigned char computed_hash[HASH_SIZE];
        compute_hash(memory[start], BLOCK_SIZE, computed_hash);
        return memcmp(computed_hash, node->hash, HASH_SIZE) == 0;
    }

    int mid = (start + end) / 2;
    int valid;
    if (index <= mid) {
        valid = verify_mit(node->left, memory, index, start, mid);
    } else {
        valid = verify_mit(node->right, memory, index, mid + 1, end);
    }

    if (!valid) return 0;

    // Recalculate the hash for the current node
    unsigned char combined_hash[2 * HASH_SIZE];
    memcpy(combined_hash, node->left->hash, HASH_SIZE);
    memcpy(combined_hash + HASH_SIZE, node->right->hash, HASH_SIZE);
    unsigned char current_hash[HASH_SIZE];
    compute_hash(combined_hash, 2 * HASH_SIZE, current_hash);

    return memcmp(current_hash, node->hash, HASH_SIZE) == 0;
}

// Function to print the MIT and memory
void print_mit(MITNode *node, int level) {
    if (node == NULL) return;

    // Indentation based on the level of the node
    for (int i = 0; i < level; i++) {
        printf("  ");
    }

    // Print the hash
    printf("Hash: ");
    for (int i = 0; i < HASH_SIZE; i++) {
        printf("%02x", node->hash[i]);
    }
    printf("\n");

    // Print left and right subtrees
    print_mit(node->left, level + 1);
    print_mit(node->right, level + 1);
}

void print_memory(unsigned char memory[][BLOCK_SIZE], int blocks) {
    for (int i = 0; i < blocks; i++) {
        printf("Memory Block %d: %s\n", i + 1, memory[i]);
    }
}

int main() {
    // Simulate memory with 4 blocks of data
    unsigned char memory[4][BLOCK_SIZE] = {
        "Block 1 data",
        "Block 2 data",
        "Block 3 data",
        "Block 4 data"
    };

    // Build the MIT
    MITNode *root = build_mit(memory, 0, 3);

    // Update a block and the MIT
    strcpy((char *)memory[1], "New Block 2 data");
    update_mit(root, memory, 1, 0, 3);

    // Verify the integrity of Block 2
    if (verify_mit(root, memory, 1, 0, 3)) {
        printf("Block 2 is intact.\n");
    } else {
        printf("Block 2 has been tampered with!\n");
    }

    // Print the MIT and memory contents
    printf("\nMemory Integrity Tree:\n");
    print_mit(root, 0);

    printf("\nMemory Contents:\n");
    print_memory(memory, 4);

    // Clean up
    free(root);

    return 0;
}

