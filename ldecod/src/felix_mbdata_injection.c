//
//  felix_mbdata_injection.c
//  ldecod
//
//  Full macroblock data injection for ML pipeline.
//

#include "felix_mbdata.h"


// hacking temp stuff
static FILE *mb_log_fp = NULL;

static inline void open_mb_log_once(void)
{
    if (!mb_log_fp)
    {
        mb_log_fp = fopen("mb_intra_debug.log", "a");  // append mode
        if (!mb_log_fp)
        {
            perror("Failed to open mb_intra_debug.log");
        }
    }
}



// Simple decode-order frame counter (matches extraction)
static int injection_frame_counter = 0;  // Monotonically increasing frame index
static int injection_last_poc = -99999;  // Last seen POC to detect frame changes
static int injection_last_logged_frame = -1;  // For debug logging

// Load MB data file into memory
void load_mbdata_file(void)
{
    FILE *f = fopen(felix_mbdata_in_path, "rb");
    if (!f) {
        printf("ERROR: Could not open %s for reading\n", felix_mbdata_in_path);
        return;
    }

    // Allocate injection context
    mbdata_i_ctx = (FelixMBDataContext*)malloc(sizeof(FelixMBDataContext));
    memset(mbdata_i_ctx, 0, sizeof(FelixMBDataContext));

    // Read header
    MBDataFileHeader header;
    size_t read_count = fread(&header, sizeof(MBDataFileHeader), 1, f);
    if (read_count != 1) {
        printf("ERROR: Failed to read header from %s\n", felix_mbdata_in_path);
        fclose(f);
        return;
    }

    // Validate magic number
    if (header.magic_number != FELIX_MBDATA_MAGIC) {
        printf("ERROR: Invalid magic number 0x%08X (expected 0x%08X)\n",
               header.magic_number, FELIX_MBDATA_MAGIC);
        fclose(f);
        return;
    }

    // Check version compatibility
    if ((header.version >> 16) != (FELIX_MBDATA_VERSION >> 16)) {
        printf("WARNING: Version mismatch - file 0x%08X, expected 0x%08X\n",
               header.version, FELIX_MBDATA_VERSION);
    }

    // Store dimensions
    mbdata_i_ctx->num_frames = header.num_frames;
    mbdata_i_ctx->capacity_frames = header.num_frames;
    mbdata_i_ctx->H_MB = header.frame_height_mbs;
    mbdata_i_ctx->W_MB = header.frame_width_mbs;
    mbdata_i_ctx->H4 = header.frame_height_mbs * 4;
    mbdata_i_ctx->W4 = header.frame_width_mbs * 4;
    mbdata_i_ctx->chroma_format_idc = header.chroma_format_idc;

    // IDR tracking - reset for injection
    mbdata_i_ctx->IDR_cnt = 0;
    mbdata_i_ctx->frames_before_last_IDR = 0;
    mbdata_i_ctx->entered_IDR_toggle = FALSE;

    // Reset static counters for frame tracking
    injection_frame_counter = 0;
    injection_last_poc = -99999;
    injection_last_logged_frame = -1;

    // Allocate and read MB data
    size_t total_mbs = (size_t)header.num_frames * header.frame_height_mbs * header.frame_width_mbs;
    mbdata_i_ctx->mb_data_field = (MacroblockData*)malloc(total_mbs * sizeof(MacroblockData));

    read_count = fread(mbdata_i_ctx->mb_data_field, sizeof(MacroblockData), total_mbs, f);
    if (read_count != total_mbs) {
        printf("ERROR: Expected %zu MBs, read %zu\n", total_mbs, read_count);
    }

    fclose(f);

    // Diagnostics
    printf("\n============= MBDATA INJECTION LOAD DIAGNOSTICS ================\n");
    printf("Input file: %s\n", felix_mbdata_in_path);
    printf("Magic: 0x%08X, Version: 0x%08X\n", header.magic_number, header.version);
    printf("Frames: %d, H_MB: %d, W_MB: %d\n",
           header.num_frames, header.frame_height_mbs, header.frame_width_mbs);
    printf("Chroma format: %d (1=4:2:0, 2=4:2:2, 3=4:4:4)\n", header.chroma_format_idc);
    printf("Bytes per MB: %d (expected %zu)\n", header.bytes_per_mb, sizeof(MacroblockData));
    printf("Total MBs loaded: %zu\n", total_mbs);
    printf("=================================================================\n\n");
}

// Inject MB metadata into currMB
static void inject_mb_metadata(Macroblock *currMB, MacroblockData *mb_data)
{
    currMB->mb_type = mb_data->metadata.mb_type;
    currMB->cbp = mb_data->metadata.cbp;
    currMB->qp = mb_data->metadata.qp;
    currMB->is_intra_block = (Boolean)mb_data->metadata.is_intra_block;

    for (int i = 0; i < 4; i++) {
        currMB->b8mode[i] = mb_data->metadata.b8mode[i];
        currMB->b8pdir[i] = mb_data->metadata.b8pdir[i];
    }

    currMB->i16mode = mb_data->metadata.i16mode;
    currMB->c_ipred_mode = mb_data->metadata.c_ipred_mode;
    currMB->luma_transform_size_8x8_flag = (Boolean)mb_data->metadata.luma_transform_size_8x8_flag;
}

// Inject motion vectors and reference indices
static void inject_block_motion(Macroblock *currMB, MacroblockData *mb_data)
{
    Slice *currSlice = currMB->p_Slice;
    StorablePicture *dec_pic = currSlice->dec_picture;
    PicMotionParams **mv_info = dec_pic->mv_info;

    int mbx_base = currMB->block_x;
    int mby_base = currMB->block_y;

    for (int by = 0; by < 4; by++) {
        for (int bx = 0; bx < 4; bx++) {
            int y4 = mby_base + by;
            int x4 = mbx_base + bx;
            int block_idx = by * 4 + bx;

            Block4x4Data *block = &mb_data->blocks[block_idx];

            // Inject MVs for both lists
            mv_info[y4][x4].mv[0] = block->mv[0];  // LIST_0 (forward)
            mv_info[y4][x4].mv[1] = block->mv[1];  // LIST_1 (backward)

            // Inject reference indices
            mv_info[y4][x4].ref_idx[0] = block->ref_idx[0];
            mv_info[y4][x4].ref_idx[1] = block->ref_idx[1];
        }
    }
}

// Inject intra prediction modes
static void inject_intra_modes(Macroblock *currMB, MacroblockData *mb_data)
{
    Slice *currSlice = currMB->p_Slice;
    byte **ipredmode = currSlice->ipredmode;

    int block_x = currMB->block_x;
    int block_y = currMB->block_y;

    // Only inject if this is an I4MB macroblock
    if (currMB->mb_type == I4MB) {
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                int y4 = block_y + by;
                int x4 = block_x + bx;
                int block_idx = by * 4 + bx;

                if (mb_data->blocks[block_idx].ipredmode >= 0) {
                    ipredmode[y4][x4] = (byte)mb_data->blocks[block_idx].ipredmode;
                }
            }
        }
    }
}

// Inject transform coefficients
static void inject_coefficients(Macroblock *currMB, MacroblockData *mb_data)
{
    Slice *currSlice = currMB->p_Slice;
    int ***cof = currSlice->cof;

    // Luma coefficients (plane 0): 16x16 block
    for (int j = 0; j < 16; j++) {
        for (int i = 0; i < 16; i++) {
            int block_idx = (j / 4) * 4 + (i / 4);
            int coef_idx = (j % 4) * 4 + (i % 4);
            cof[0][j][i] = mb_data->luma_cof[block_idx][coef_idx];
        }
    }

    // Chroma coefficients for 4:2:0 (planes 1 and 2, each 8x8)
    if (mbdata_i_ctx->chroma_format_idc > 0) {
        for (int plane = 1; plane <= 2; plane++) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < 8; i++) {
                    int coef_idx = j * 8 + i;
                    cof[plane][j][i] = mb_data->chroma_cof[plane - 1][coef_idx];
                }
            }
        }
    }

    // Signal that we have injected coefficients (prevent reset)
    currSlice->is_reset_coeff = FALSE;
    currSlice->is_reset_coeff_cr = FALSE;
}

// Main injection function - called after read_one_macroblock(), before decode_one_macroblock()
void inject_mbdata(Macroblock *currMB)
{
    if (!mbdata_i_ctx || !mbdata_i_ctx->mb_data_field) {
        printf("ERROR: inject_mbdata called but no data loaded\n");
        return;
    }

    Slice *currSlice = currMB->p_Slice;
    StorablePicture *dec_pic = currSlice->dec_picture;
    int is_idr = dec_pic->idr_flag;
    int current_poc = dec_pic->frame_poc;

    // Detect new frame: POC changed from last macroblock
    if (current_poc != injection_last_poc) {
        // New frame - increment if this is not the very first MB
        if (injection_last_poc != -99999) {
            injection_frame_counter++;
        }
        injection_last_poc = current_poc;
    }

    int f = injection_frame_counter;

    // Debug: log once per frame
    if (f != injection_last_logged_frame) {
        printf("Inject frame %d: POC=%d, idr=%d\n", f, current_poc, is_idr);
        injection_last_logged_frame = f;
    }

    // Get MB coordinates
    int mb_x = currMB->mb.x;
    int mb_y = currMB->mb.y;

    // Bounds check
    if (f >= mbdata_i_ctx->capacity_frames) {
        printf("WARNING: Frame %d exceeds loaded data (%d frames)\n",
               f, mbdata_i_ctx->capacity_frames);
        return;
    }

    // Get pointer to MB data
    MacroblockData *mb_data = mbdata_at(mbdata_i_ctx, f, mb_y, mb_x);
    if (!mb_data) {
        printf("ERROR: mbdata_at returned NULL for f=%d, mb_y=%d, mb_x=%d\n", f, mb_y, mb_x);
        return;
    }

    // Inject motion, intra modes, and coefficients
    // NOTE: We do NOT inject metadata (mb_type, b8mode, b8pdir, etc.) because
    // the decoder has already used these to build reference picture lists.
    // Changing them after parsing causes "invalid reference picture" errors.
    inject_block_motion(currMB, mb_data);
    inject_intra_modes(currMB, mb_data);
    inject_coefficients(currMB, mb_data);
  
    /* --------------------------------------------------
     * Force CBP for non-intra macroblocks
     * -------------------------------------------------- */
    
    open_mb_log_once();

    if (mb_log_fp)
    {
        fprintf(
            mb_log_fp,
            "[MB addr=%6d pix=(%4d,%4d) blk=(%4d,%4d)] "
            "mb_type=%d  JM_is_intra=%d  meta_is_intra=%d  cbp=0x%02X  tr8x8=%d\n",
            currMB->mbAddrX,
            currMB->pix_x, currMB->pix_y,
            currMB->block_x, currMB->block_y,
            currMB->mb_type,
            currMB->is_intra_block ? 1 : 0,
            mb_data->metadata.is_intra_block,
            currMB->cbp,
            currMB->luma_transform_size_8x8_flag ? 1 : 0
        );

        fflush(mb_log_fp);  // ensure data is written immediately
    }

  
    if (mb_data->metadata.is_intra_block == 0)
    {
      currMB->cbp = mb_data->metadata.cbp;
      // instead of inject_metadata, we just make sure all luma and chroma are read from injection, instead of copied. This check is performed in block.c.
    }

    // Update ref_pic pointers to match injected ref_idx values
    // (read_motion_info_from_NAL sets ref_pic based on original ref_idx,
    //  so we must update them after injection changes ref_idx)
    int list_offset = currMB->list_offset;
    StorablePicture **list0 = currSlice->listX[LIST_0 + list_offset];
    StorablePicture **list1 = currSlice->listX[LIST_1 + list_offset];
    PicMotionParams **mv_info = dec_pic->mv_info;

    for (int by = 0; by < 4; by++) {
        for (int bx = 0; bx < 4; bx++) {
            int y4 = currMB->block_y + by;
            int x4 = currMB->block_x + bx;

            short ref0 = mv_info[y4][x4].ref_idx[0];
            short ref1 = mv_info[y4][x4].ref_idx[1];

            if (ref0 >= 0)
                mv_info[y4][x4].ref_pic[LIST_0] = list0[ref0];
            else
                mv_info[y4][x4].ref_pic[LIST_0] = NULL;

            if (ref1 >= 0)
                mv_info[y4][x4].ref_pic[LIST_1] = list1[ref1];
            else
                mv_info[y4][x4].ref_pic[LIST_1] = NULL;
        }
    }
}

// Cleanup injection context
void free_mbdata_injection(void)
{
    if (mbdata_i_ctx) {
        if (mbdata_i_ctx->mb_data_field) {
            free(mbdata_i_ctx->mb_data_field);
            mbdata_i_ctx->mb_data_field = NULL;
        }
        free(mbdata_i_ctx);
        mbdata_i_ctx = NULL;
    }
}
