#pragma once
#include <cstdint>

struct IctcMapParams {
    uint64_t nodes;
    uint32_t channels, right_channels, left_width, right_width, paths, grad_width;
    uint32_t output_width, mode, bank, indexed, tile_rows, tile_channels;
    uint32_t terms, head, global_term_stride, local_term_stride;
    uint64_t output_nodes;
    uint32_t graph, masked;
};
