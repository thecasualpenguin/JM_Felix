//
//  felix_mbdata_extraction.c
//  ldecod
//
//  Full macroblock data extraction for ML pipeline.
//

#include "felix_mbdata.h"

// Initialize MB data extraction once width/height are known.
void init_mbdata_extraction(VideoParameters *p_Vid)
{
    mbdata_e_ctx = (FelixMBDataContext*)malloc(sizeof(FelixMBDataContext));
    memset(mbdata_e_ctx, 0, sizeof(FelixMBDataContext));

    // Dimensions in MBs (each MB is 16x16 pixels)
    mbdata_e_ctx->H_MB = p_Vid->height / 16;
    mbdata_e_ctx->W_MB = p_Vid->width / 16;

    // Also store 4x4 block dimensions for reference
    mbdata_e_ctx->H4 = p_Vid->height / 4;
    mbdata_e_ctx->W4 = p_Vid->width / 4;

    mbdata_e_ctx->capacity_frames = 16;  // Start with small capacity; will grow as needed
    mbdata_e_ctx->num_frames = 0;

    // IDR tracking
    mbdata_e_ctx->IDR_cnt = 0;
    mbdata_e_ctx->frames_before_last_IDR = 0;
    mbdata_e_ctx->entered_IDR_toggle = FALSE;

    // Chroma format (typically 4:2:0 = 1)
    mbdata_e_ctx->chroma_format_idc = p_Vid->yuv_format;

    // Allocate MB data buffer
    size_t total_mbs = (size_t)mbdata_e_ctx->capacity_frames * mbdata_e_ctx->H_MB * mbdata_e_ctx->W_MB;
    mbdata_e_ctx->mb_data_field = (MacroblockData*)malloc(total_mbs * sizeof(MacroblockData));
    memset(mbdata_e_ctx->mb_data_field, 0, total_mbs * sizeof(MacroblockData));

    printf("init_mbdata_extraction: H_MB=%d, W_MB=%d, initial capacity=%d frames\n",
           mbdata_e_ctx->H_MB, mbdata_e_ctx->W_MB, mbdata_e_ctx->capacity_frames);
}

// Ensure we have space for frame index f (0-based).
void ensure_mbdata_frame_capacity(int f)
{
    if (f < mbdata_e_ctx->capacity_frames)
        return;

    int32_t new_cap = mbdata_e_ctx->capacity_frames;
    while (f >= new_cap)
        new_cap *= 2;

    size_t old_total = (size_t)mbdata_e_ctx->capacity_frames * mbdata_e_ctx->H_MB * mbdata_e_ctx->W_MB;
    size_t new_total = (size_t)new_cap * mbdata_e_ctx->H_MB * mbdata_e_ctx->W_MB;

    mbdata_e_ctx->mb_data_field = (MacroblockData*)realloc(
        mbdata_e_ctx->mb_data_field,
        new_total * sizeof(MacroblockData)
    );

    // Zero the newly allocated area
    memset(mbdata_e_ctx->mb_data_field + old_total, 0,
           (new_total - old_total) * sizeof(MacroblockData));

    mbdata_e_ctx->capacity_frames = new_cap;

    printf("ensure_mbdata_frame_capacity: expanded to %d frames (%zu MBs)\n", new_cap, new_total);
}

// Extract MB metadata
static void extract_mb_metadata(Macroblock *currMB, MacroblockData *mb_data)
{
    mb_data->metadata.mb_type = currMB->mb_type;
    mb_data->metadata.cbp = (int16_t)currMB->cbp;
    mb_data->metadata.qp = (int8_t)currMB->qp;
    mb_data->metadata.is_intra_block = (int8_t)currMB->is_intra_block;

    for (int i = 0; i < 4; i++) {
        mb_data->metadata.b8mode[i] = currMB->b8mode[i];
        mb_data->metadata.b8pdir[i] = currMB->b8pdir[i];
    }

    mb_data->metadata.i16mode = (int8_t)currMB->i16mode;
    mb_data->metadata.c_ipred_mode = currMB->c_ipred_mode;
    mb_data->metadata.luma_transform_size_8x8_flag = (int8_t)currMB->luma_transform_size_8x8_flag;
    mb_data->metadata.reserved = 0;
}

// Extract per-4x4 block motion info (MVs and ref indices)
static void extract_block_motion(Macroblock *currMB, MacroblockData *mb_data)
{
    Slice *currSlice = currMB->p_Slice;
    StorablePicture *dec_pic = currSlice->dec_picture;
    PicMotionParams **mv_info = dec_pic->mv_info;

    int mbx_base = currMB->block_x;  // Top-left 4x4 coord of MB (X)
    int mby_base = currMB->block_y;  // Top-left 4x4 coord of MB (Y)

    for (int by = 0; by < 4; by++) {
        for (int bx = 0; bx < 4; bx++) {
            int y4 = mby_base + by;
            int x4 = mbx_base + bx;
            int block_idx = by * 4 + bx;

            Block4x4Data *block = &mb_data->blocks[block_idx];

            // Extract MVs for both lists
            block->mv[0] = mv_info[y4][x4].mv[0];  // LIST_0 (forward)
            block->mv[1] = mv_info[y4][x4].mv[1];  // LIST_1 (backward)

            // Extract reference indices
            block->ref_idx[0] = mv_info[y4][x4].ref_idx[0];
            block->ref_idx[1] = mv_info[y4][x4].ref_idx[1];
        }
    }
}

// Extract intra prediction modes for 4x4 blocks
static void extract_intra_modes(Macroblock *currMB, MacroblockData *mb_data)
{
    Slice *currSlice = currMB->p_Slice;
    byte **ipredmode = currSlice->ipredmode;

    int block_x = currMB->block_x;
    int block_y = currMB->block_y;

    for (int by = 0; by < 4; by++) {
        for (int bx = 0; bx < 4; bx++) {
            int y4 = block_y + by;
            int x4 = block_x + bx;
            int block_idx = by * 4 + bx;

            // Only I4MB has meaningful per-block intra modes
            if (currMB->mb_type == I4MB) {
                mb_data->blocks[block_idx].ipredmode = (int8_t)ipredmode[y4][x4];
            } else {
                mb_data->blocks[block_idx].ipredmode = -1;  // Not applicable
            }
            mb_data->blocks[block_idx].reserved = 0;
        }
    }
}

// Extract transform coefficients
static void extract_coefficients(Macroblock *currMB, MacroblockData *mb_data)
{
    Slice *currSlice = currMB->p_Slice;
    int ***cof = currSlice->cof;  // cof[plane][y][x]

    // Luma coefficients (plane 0): 16x16 block
    for (int j = 0; j < 16; j++) {
        for (int i = 0; i < 16; i++) {
            // Map to 4x4 block structure
            int block_idx = (j / 4) * 4 + (i / 4);
            int coef_idx = (j % 4) * 4 + (i % 4);
            mb_data->luma_cof[block_idx][coef_idx] = (int16_t)cof[0][j][i];
        }
    }

    // Chroma coefficients for 4:2:0 (planes 1 and 2, each 8x8)
    // Only extract if chroma is present
    if (mbdata_e_ctx->chroma_format_idc > 0) {
        for (int plane = 1; plane <= 2; plane++) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < 8; i++) {
                    int coef_idx = j * 8 + i;
                    mb_data->chroma_cof[plane - 1][coef_idx] = (int16_t)cof[plane][j][i];
                }
            }
        }
    }
}

// Main extraction function - called after read_one_macroblock()
void extract_mbdata(Macroblock *currMB)
{
    Slice *currSlice = currMB->p_Slice;
    VideoParameters *p_Vid = currMB->p_Vid;
    StorablePicture *dec_pic = currSlice->dec_picture;
    int is_idr = dec_pic->idr_flag;

    // Initialize on first call
    if (!mbdata_e_ctx)
        init_mbdata_extraction(p_Vid);

    // Track IDR frame-no resets
    if (is_idr && !mbdata_e_ctx->entered_IDR_toggle) {
        mbdata_e_ctx->entered_IDR_toggle = TRUE;
        mbdata_e_ctx->frames_before_last_IDR = mbdata_e_ctx->num_frames;
        ++mbdata_e_ctx->IDR_cnt;
    }
    if (!is_idr)
        mbdata_e_ctx->entered_IDR_toggle = FALSE;

    // Current frame index (0-based)
    int f = dec_pic->frame_poc / 2 + mbdata_e_ctx->frames_before_last_IDR;

    // Ensure capacity
    ensure_mbdata_frame_capacity(f);

    // Track highest frame index
    if (f + 1 > mbdata_e_ctx->num_frames)
        mbdata_e_ctx->num_frames = f + 1;

    // Get MB coordinates
    int mb_x = currMB->mb.x;
    int mb_y = currMB->mb.y;

    // Get pointer to MB data slot
    MacroblockData *mb_data = mbdata_at(mbdata_e_ctx, f, mb_y, mb_x);
    if (!mb_data) {
        printf("ERROR: mbdata_at returned NULL for f=%d, mb_y=%d, mb_x=%d\n", f, mb_y, mb_x);
        return;
    }

    // Extract all MB data
    extract_mb_metadata(currMB, mb_data);
    extract_block_motion(currMB, mb_data);
    extract_intra_modes(currMB, mb_data);
    extract_coefficients(currMB, mb_data);
}

// Write complete MB data to binary file
void write_mbdata_file(void)
{
    if (!mbdata_e_ctx) {
        printf("write_mbdata_file: No extraction context\n");
        return;
    }

    FILE *f = fopen(felix_mbdata_out_path, "wb");
    if (!f) {
        printf("ERROR: Could not open %s for writing\n", felix_mbdata_out_path);
        return;
    }

    // Write header
    MBDataFileHeader header;
    header.magic_number = FELIX_MBDATA_MAGIC;
    header.version = FELIX_MBDATA_VERSION;
    header.num_frames = mbdata_e_ctx->num_frames;
    header.frame_height_mbs = mbdata_e_ctx->H_MB;
    header.frame_width_mbs = mbdata_e_ctx->W_MB;
    header.chroma_format_idc = mbdata_e_ctx->chroma_format_idc;
    header.bytes_per_mb = sizeof(MacroblockData);
    header.reserved = 0;

    fwrite(&header, sizeof(MBDataFileHeader), 1, f);

    // Write MB data
    size_t total_mbs = (size_t)mbdata_e_ctx->num_frames * mbdata_e_ctx->H_MB * mbdata_e_ctx->W_MB;
    fwrite(mbdata_e_ctx->mb_data_field, sizeof(MacroblockData), total_mbs, f);

    // Diagnostics
    printf("\n============= MBDATA EXTRACTION DIAGNOSTICS ================\n");
    printf("Output file: %s\n", felix_mbdata_out_path);
    printf("Magic: 0x%08X, Version: 0x%08X\n", header.magic_number, header.version);
    printf("Frames: %d, H_MB: %d, W_MB: %d\n",
           header.num_frames, header.frame_height_mbs, header.frame_width_mbs);
    printf("Chroma format: %d (1=4:2:0, 2=4:2:2, 3=4:4:4)\n", header.chroma_format_idc);
    printf("Bytes per MB: %d\n", header.bytes_per_mb);
    printf("Total MBs: %zu\n", total_mbs);
    printf("Header size: %zu bytes\n", sizeof(MBDataFileHeader));
    printf("Data size: %zu bytes\n", total_mbs * sizeof(MacroblockData));
    printf("Total file size: %zu bytes\n", sizeof(MBDataFileHeader) + total_mbs * sizeof(MacroblockData));

    // Count some statistics
    int intra_count = 0, inter_count = 0, nonzero_mv_count = 0, nonzero_coef_count = 0;
    for (size_t i = 0; i < total_mbs; i++) {
        MacroblockData *mb = &mbdata_e_ctx->mb_data_field[i];
        if (mb->metadata.is_intra_block)
            intra_count++;
        else
            inter_count++;

        for (int b = 0; b < 16; b++) {
            if (mb->blocks[b].mv[0].mv_x != 0 || mb->blocks[b].mv[0].mv_y != 0)
                nonzero_mv_count++;
            if (mb->blocks[b].mv[1].mv_x != 0 || mb->blocks[b].mv[1].mv_y != 0)
                nonzero_mv_count++;
        }

        for (int b = 0; b < 16; b++) {
            for (int c = 0; c < 16; c++) {
                if (mb->luma_cof[b][c] != 0)
                    nonzero_coef_count++;
            }
        }
    }
    printf("Intra MBs: %d, Inter MBs: %d\n", intra_count, inter_count);
    printf("Non-zero MVs (both lists): %d\n", nonzero_mv_count);
    printf("Non-zero luma coefficients: %d\n", nonzero_coef_count);
    printf("=============================================================\n\n");

    fclose(f);
}

// Cleanup extraction context
void free_mbdata_extraction(void)
{
    if (mbdata_e_ctx) {
        if (mbdata_e_ctx->mb_data_field) {
            free(mbdata_e_ctx->mb_data_field);
            mbdata_e_ctx->mb_data_field = NULL;
        }
        free(mbdata_e_ctx);
        mbdata_e_ctx = NULL;
    }
}
