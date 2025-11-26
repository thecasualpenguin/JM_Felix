//
//  felix.c
//  ldecod
//
//  Created by Felix Gan on 11/17/25.
//

#include "felix.h"

// Allocate and initialize
FelixVideoContext *create_felix_context(void)
{
    FelixVideoContext *ctx = (FelixVideoContext*)malloc(sizeof(FelixVideoContext));
    if (!ctx) return NULL;

    ctx->g_num_frames      = 0;
    ctx->g_capacity_frames = 0;
    ctx->g_H4              = 0;
    ctx->g_W4              = 0;
    ctx->g_mv_field        = NULL;

    ctx->IDR_cnt = 0;     // idr refreshes reset frame-no, so keep track
    ctx->frames_before_last_IDR = 0;
    ctx->entered_IDR_toggle = FALSE;    // prevents MB

    return ctx;
}

/*
 * load_mv_file:
 *   Loads a binary motion vector file created by convert_mvs.py.
 *
 * File format (little-endian):
 *   int32  num_frames   (F)
 *   int32  height4      (H4)
 *   int32  width4       (W4)
 *   Then F * H4 * W4 * 2 int16 values:
 *       mv[f][y][x][0] = mvx
 *       mv[f][y][x][1] = mvy
 *
 * We store these into a single flat array of CustomMV:
 *   g_mv_field[idx] where:
 *       idx = ((f * H4) + y) * W4 + x
 */
void load_mv_file(void)
{
    FILE *f = fopen(felix_mv_in_path, "rb");
    i_ctx = create_felix_context();

    // --- Read the 3 header integers ---
    fread(&i_ctx->g_num_frames, sizeof(int32_t), 1, f);
    fread(&i_ctx->g_H4,        sizeof(int32_t), 1, f);
    fread(&i_ctx->g_W4,        sizeof(int32_t), 1, f);

    // --- Compute how many CustomMV structs we need ---
    size_t total = (size_t)i_ctx->g_num_frames * i_ctx->g_H4 * i_ctx->g_W4;

    // --- Allocate memory to hold all MV pairs ---
    i_ctx->g_mv_field = malloc(total * sizeof(MotionVector));

    // --- Read F*H4*W4*(mvx,mvy) from file into memory ---
    fread(i_ctx->g_mv_field, sizeof(MotionVector), total, f);
    fclose(f);
  
    // RESET g_num_frames because we need it as a running counter to track frames before IDR resets
    i_ctx->g_num_frames = 0;
  
    // -------------------- DIAGNOSTICS START --------------------
    printf("\n============== INJECTION DIAGNOSTICS ===============\n");
    printf("File pointer (before fclose): %p\n", (void*)f);
    printf("Total MotionVectors: %zu\n", total);

    int nonzero_count = 0;

    for (size_t i = 0; i < total; i++) {
        if (i_ctx->g_mv_field[i].mv_x != 0 || i_ctx->g_mv_field[i].mv_y != 0) {
            nonzero_count++;

            // Print first few non-zero MV entries
            if (nonzero_count <= 10) {
                printf("  Non-zero MV at index %zu: (mvx=%d, mvy=%d)\n",
                       i,
                       i_ctx->g_mv_field[i].mv_x,
                       i_ctx->g_mv_field[i].mv_y);
            }
        }
    }

    printf("Total non-zero MotionVectors: %d\n", nonzero_count);
    printf("=====================================================\n\n");
    // --------------------- DIAGNOSTICS END ---------------------

}

/*
 * mv_at:
 *   Return pointer to the MV for:
 *       frame index f
 *       4x4 block coordinate (y4, x4)
 *
 *   Mapping:
 *      data[f][y4][x4]  → idx = ((f * H4) + y4) * W4 + x4
 *
 *   The caller can then read:
 *      mv->mv_x, mv->mv_y
 */
static inline const MotionVector *mv_at(int f, int y4, int x4)
{
    // ----- Bound checks -----
    if (f   < 0 || f   >= i_ctx->g_num_frames ||
        y4  < 0 || y4  >= i_ctx->g_H4         ||
        x4  < 0 || x4  >= i_ctx->g_W4) {
      return &zero_mv;
    }
  
//    fprintf(stdout, "HIT f=%d y4=%d x4=%d\n", f, y4, x4);
  
    // ----- Safe index computation -----
    size_t idx = ((size_t)f * i_ctx->g_H4 + y4) * i_ctx->g_W4 + x4;
    return &i_ctx->g_mv_field[idx];
}



void sanity_check_block_multiple(void)
{
    printf("BLOCK_MULTIPLE = %d, sizeof(mvd) = %zu\n",
           BLOCK_MULTIPLE,
           sizeof(((Macroblock*)0)->mvd));
}


// call once, e.g. right after parsing config / before decoding
void open_mv_log(void)
{
  if (!mv_log) {
    mv_log = fopen(mv_log_path, "w");
    if (!mv_log) {
      perror("fopen mvs.log");
      exit(1);
    }
  }
}

// call once at the end of decoding
void close_mv_log(void)
{
  if (mv_log) {
    fclose(mv_log);
    mv_log = NULL;
  }
}

void print_mvs(Macroblock *currMB)
{
  int list, by, bx;
  
  Slice *currSlice = currMB->p_Slice;
  int slice_type   = currSlice->slice_type;
  VideoParameters *p_Vid     = currMB->p_Vid;
  StorablePicture *dec_pic   = currSlice->dec_picture;
  PicMotionParams **mv_info  = dec_pic->mv_info;
  
  int mbx_base = currMB->block_x; // 4x4 grid coords
  int mby_base = currMB->block_y;


  for (by = 0; by < BLOCK_MULTIPLE; ++by)
  {
    for (bx = 0; bx < BLOCK_MULTIPLE; ++bx)
    {
      for (list = 0; list < 2; ++list) 
      /* Two lists in JM: 0 = LIST_0 (forward), 1 = LIST_1 (backward) */
      {
        fprintf(mv_log, 
                "width:               %d\n"
                "height:              %d\n"
                "max_frame_num:       %d\n"
                "PicWidthInMbs:       %u\n"
                "PicHeightInMapUnits: %u\n"
                "FrameHeightInMbs:    %u\n"
                "FrameSizeInMbs:      %u\n"
                "width_cr:            %d\n"
                "height_cr:           %d\n"
                "frame_no:            %d\n"
                "number:              %d\n",
                p_Vid->width,
                p_Vid->height,
                p_Vid->max_frame_num,
                p_Vid->PicWidthInMbs,
                p_Vid->PicHeightInMapUnits,
                p_Vid->FrameHeightInMbs,
                p_Vid->FrameSizeInMbs,
                p_Vid->width_cr,
                p_Vid->height_cr,
                p_Vid->frame_no,
                p_Vid->number);

        fprintf(mv_log,
                "Frame %04d MB %d list %d @ [%d,%d] mv = (%d,%d)\n",
                p_Vid->frame_no,
                currMB->mbAddrX, list, by, bx,
                mv_info[mby_base + by][mbx_base + bx].mv[list].mv_x,
                mv_info[mby_base + by][mbx_base + bx].mv[list].mv_y
                );
      }
    }
  }
}


void apply_custom_mvs_to_mb(Macroblock *currMB)
{
  int list, by, bx;
  const MotionVector *cmv;
  MotionVector *origMV;
  
  Slice *currSlice = currMB->p_Slice;
  VideoParameters *p_Vid     = currMB->p_Vid;
  StorablePicture *dec_pic   = currSlice->dec_picture;
  PicMotionParams **mv_info  = dec_pic->mv_info;
  int mbx_base = currMB->block_x; // 4x4 grid coords (top left x and y)
  int mby_base = currMB->block_y;
  int is_idr     = dec_pic->idr_flag;

  // track idr frame-no resets
  if (is_idr && !i_ctx->entered_IDR_toggle) {
    i_ctx->entered_IDR_toggle = TRUE;
    i_ctx->frames_before_last_IDR = i_ctx->g_num_frames;
    ++i_ctx->IDR_cnt;
  }
  if (!is_idr) i_ctx->entered_IDR_toggle = FALSE;
  
  // find globally consistent frame number
  int f = dec_pic->frame_poc/2 + i_ctx->frames_before_last_IDR;
  
  // Track the highest frame index we've seen so far.
  if (f + 1 > i_ctx->g_num_frames)
      i_ctx->g_num_frames = f + 1;
  
  for (by = 0; by < BLOCK_MULTIPLE; ++by) {
    for (bx = 0; bx < BLOCK_MULTIPLE; ++bx) {
      for (list = 0; list < 2; ++list) {
        /* Two lists in JM: 0 = LIST_0 (forward), 1 = LIST_1 (backward) */
//        if (list==0) continue;
        int y4 = mby_base + by;
        int x4 = mbx_base + bx;
        if (list==0)
        {
          cmv = mv_at(f, y4, x4);
          origMV = &mv_info[y4][x4].mv[list];
          *origMV = *cmv;
        }
        
        if (list==1)
        {
          origMV = &mv_info[y4][x4].mv[list];
          *origMV = zero_mv;
        }
        
        
//        if (cmv->mv_x != 0 || cmv->mv_y != 0){
//          fprintf(stdout, "valid = %s, curr= (%d, %d), new = (%d, %d) \n",
//                  ( ((*origMV).mv_x == cmv->mv_x) && ((*origMV).mv_y == cmv->mv_y)) ?"true":"false",
//                  origMV->mv_x, origMV->mv_y,
//                  cmv->mv_x, cmv->mv_y);
//        }
       

        
        
      }
    }
  }
}


