#pragma once
#include <cstdint>
#include "acl/acl.h"

// Host-callable launchers compiled by the CANN ASC compiler.
namespace ictc_ascendc {
void launch_batched_vector(uint32_t blocks, aclrtStream stream,
    uint8_t* a, uint8_t* b, uint8_t* out, uint8_t* ix, uint64_t n, uint64_t as, uint64_t bs,
    uint32_t q, uint32_t i, uint32_t c, uint32_t mode, uint32_t rows, uint32_t bc);
void launch_epilogue(uint32_t blocks, aclrtStream stream,
    uint8_t* l, uint8_t* r, uint8_t* g, uint8_t* dst, uint8_t* mask, uint8_t* meta, uint8_t* ix,
    uint8_t* out, uint64_t n, uint64_t ls, uint64_t rs, uint32_t c, uint32_t q, uint32_t p,
    uint32_t m, uint32_t paths, uint32_t mode, uint32_t rows, uint32_t bc, uint32_t masked,
    uint32_t dense, uint32_t hoist);
void launch_epilogue_adjoint(uint32_t blocks, aclrtStream stream,
    uint8_t* l, uint8_t* r, uint8_t* g, uint8_t* hl, uint8_t* hr, uint8_t* dst, uint8_t* mask,
    uint8_t* meta, uint8_t* ix, uint8_t* dl, uint8_t* dr, uint8_t* dg, uint64_t n, uint64_t ls,
    uint64_t rs, uint64_t hls, uint64_t hrs, uint32_t c, uint32_t q, uint32_t p, uint32_t m,
    uint32_t second, uint32_t rows, uint32_t bc, uint32_t masked, uint32_t flags, uint32_t hoist);
void launch_matmul(uint32_t blocks, aclrtStream stream,
    uint8_t* a, uint8_t* b, uint8_t* bias, uint8_t* c, uint8_t* workspace, uint8_t* metadata,
    uint32_t transposeA, uint32_t transposeB, uint32_t hasBias);
void launch_binary(uint32_t blocks, aclrtStream stream,
    uint8_t* x, uint8_t* y, uint8_t* z, uint64_t size, uint32_t tile, uint32_t mode, float alpha,
    float beta, uint8_t* indices, uint64_t columns);
void launch_weighted_map(uint32_t blocks, aclrtStream stream,
    uint8_t* l, uint8_t* r, uint8_t* w, uint8_t* g, uint8_t* species, uint8_t* meta, uint8_t* ix0,
    uint8_t* ix1, uint8_t* source, uint8_t* destination, uint8_t* mask, uint8_t* out, uint64_t n,
    uint32_t c, uint32_t rc, uint32_t a, uint32_t b, uint32_t paths, uint32_t m, uint32_t o,
    uint32_t mode, uint32_t bank, uint32_t indexed, uint32_t rows, uint32_t channels,
    uint32_t terms, uint32_t head, uint32_t globalStride, uint32_t localStride,
    uint64_t outputNodes, uint32_t graph, uint32_t masked);
} // namespace ictc_ascendc
