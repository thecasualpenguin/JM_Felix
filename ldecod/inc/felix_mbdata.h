//
//  felix_mbdata.h
//  ldecod
//
//  Full macroblock data extraction and injection for ML pipeline.
//  Extends the original felix MV-only system to include all MB data:
//  - MB metadata (type, CBP, QP, prediction modes)
//  - Per 4x4 block motion vectors (both LIST_0 and LIST_1)
//  - Reference indices
//  - Intra prediction modes
//  - Transform coefficients (luma and chroma)
//

#ifndef felix_mbdata_h
#define felix_mbdata_h

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "global.h"
#include "mbuffer.h"

// File format magic number and version
#define FELIX_MBDATA_MAGIC   0x464D4232  // "FMB2"
#define FELIX_MBDATA_VERSION 0x00020000  // Version 2.0

#pragma pack(push, 1)  // Ensure tight packing for binary file I/O

// Per-4x4 block data (16 of these per MB)
typedef struct {
    MotionVector mv[2];       // mv[0] = LIST_0, mv[1] = LIST_1 (4 bytes each = 8 bytes)
    int8_t       ref_idx[2];  // Reference indices for LIST_0 and LIST_1 (2 bytes)
    int8_t       ipredmode;   // 4x4 intra prediction mode (1 byte), -1 for inter blocks
    int8_t       reserved;    // Alignment padding (1 byte)
} Block4x4Data;  // Total: 12 bytes per 4x4 block

// Per-macroblock metadata
typedef struct {
    int16_t  mb_type;           // P16x16, I4MB, I16MB, P8x8, BSKIP_DIRECT, etc.
    int16_t  cbp;               // Coded Block Pattern (lower 16 bits)
    int8_t   qp;                // Quantization Parameter (0-51)
    int8_t   is_intra_block;    // Boolean: 1 if intra, 0 if inter
    int8_t   b8mode[4];         // 8x8 block partition modes
    int8_t   b8pdir[4];         // Prediction directions (LIST_0, LIST_1, BI_PRED)
    int8_t   i16mode;           // 16x16 intra prediction mode (for I16MB, -1 otherwise)
    int8_t   c_ipred_mode;      // Chroma intra prediction mode
    int8_t   luma_transform_size_8x8_flag;  // 0: 4x4 transform, 1: 8x8 transform
    int8_t   reserved;          // Alignment padding
} MBMetadata;  // Total: 16 bytes

// Complete macroblock data for extraction/injection
typedef struct {
    MBMetadata metadata;                    // 16 bytes
    Block4x4Data blocks[16];                // 12 * 16 = 192 bytes (4x4 grid: [by*4+bx])

    // Luma coefficients: 16x16 = 256 coefficients
    // For 4x4 transform: 16 blocks of 16 coefficients each
    // For 8x8 transform: 4 blocks of 64 coefficients each (stored as 16 blocks for uniformity)
    // Layout: luma_cof[block_idx][coef_idx] where block_idx = (j/4)*4 + (i/4), coef_idx = (j%4)*4 + (i%4)
    int16_t luma_cof[16][16];               // 512 bytes

    // Chroma coefficients for 4:2:0: 8x8 per plane = 64 coefficients per plane
    // U (Cb): 4 4x4 blocks = 64 coefficients
    // V (Cr): 4 4x4 blocks = 64 coefficients
    int16_t chroma_cof[2][64];              // 256 bytes (2 planes * 64 * 2 bytes)

} MacroblockData;  // Total: 976 bytes per MB

// Binary file header
typedef struct {
    uint32_t magic_number;      // FELIX_MBDATA_MAGIC
    uint32_t version;           // FELIX_MBDATA_VERSION
    int32_t  num_frames;        // Number of frames
    int32_t  frame_height_mbs;  // Height in MBs
    int32_t  frame_width_mbs;   // Width in MBs
    int32_t  chroma_format_idc; // 1 for 4:2:0, 2 for 4:2:2, 3 for 4:4:4
    int32_t  bytes_per_mb;      // sizeof(MacroblockData)
    int32_t  reserved;          // Reserved for future use
} MBDataFileHeader;  // Total: 32 bytes

#pragma pack(pop)

// Full context for extraction/injection
typedef struct {
    // Dimension info
    int32_t num_frames;         // Actual number of frames filled
    int32_t capacity_frames;    // Allocated frames (grows dynamically)
    int32_t H_MB;               // Height in MBs
    int32_t W_MB;               // Width in MBs
    int32_t H4;                 // Height in 4x4 blocks (for reference)
    int32_t W4;                 // Width in 4x4 blocks (for reference)

    // Per-MB data array: [frame][mb_y * W_MB + mb_x]
    MacroblockData *mb_data_field;

    // IDR tracking (same as existing)
    int32_t IDR_cnt;
    int32_t frames_before_last_IDR;
    Boolean entered_IDR_toggle;

    // Chroma format info
    int32_t chroma_format_idc;
} FelixMBDataContext;

// Global contexts for extraction and injection
static FelixMBDataContext *mbdata_e_ctx = NULL;  // extraction context
static FelixMBDataContext *mbdata_i_ctx = NULL;  // injection context

// ========================= Extraction Functions =========================

// Initialize MB data extraction context
void init_mbdata_extraction(VideoParameters *p_Vid);

// Ensure we have capacity for frame f
void ensure_mbdata_frame_capacity(int f);

// Get pointer to MB data for given frame/MB coordinates
static inline MacroblockData *mbdata_at(FelixMBDataContext *ctx, int f, int mb_y, int mb_x);

// Main extraction function - called after read_one_macroblock()
void extract_mbdata(Macroblock *currMB);

// Write complete MB data to binary file
void write_mbdata_file(void);

// Cleanup extraction context
void free_mbdata_extraction(void);

// ========================= Injection Functions =========================

// Load MB data file into memory
void load_mbdata_file(void);

// Main injection function - called after read_one_macroblock(), before decode_one_macroblock()
void inject_mbdata(Macroblock *currMB);

// Cleanup injection context
void free_mbdata_injection(void);

// ========================= Inline Accessor =========================

static inline MacroblockData *mbdata_at(FelixMBDataContext *ctx, int f, int mb_y, int mb_x)
{
    if (!ctx || !ctx->mb_data_field)
        return NULL;
    if (f < 0 || mb_y < 0 || mb_x < 0)
        return NULL;
    if (f >= ctx->capacity_frames)
        return NULL;
    if (mb_y >= ctx->H_MB || mb_x >= ctx->W_MB)
        return NULL;

    size_t idx = ((size_t)f * ctx->H_MB + mb_y) * ctx->W_MB + mb_x;
    return &ctx->mb_data_field[idx];
}

#endif /* felix_mbdata_h */
