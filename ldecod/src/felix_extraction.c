//
//  felix_extraction.c
//  ldecod
//
//  Created by Felix Gan on 11/24/25.
//

#include "felix_extraction.h"

/* Initialize MV extraction once width/height are known.
 * p_Vid->width, p_Vid->height are luma width/height in pixels.
 * H4 = height / 4, W4 = width / 4, since each block is 4x4 pixels.
 */
void init_mv_extraction(VideoParameters *p_Vid)
{
    e_ctx = create_felix_context();
  
    e_ctx->g_H4 = p_Vid->height / 4;
    e_ctx->g_W4 = p_Vid->width  / 4;

    e_ctx->g_capacity_frames = 16;  // start with some small capacity; will grow as needed
    e_ctx->g_num_frames      = 0;
  
    e_ctx->IDR_cnt = 0;     // idr refreshes reset frame-no, so keep track
    e_ctx->frames_before_last_IDR = 0;
    e_ctx->entered_IDR_toggle = FALSE;    // prevents MB

    size_t total = (size_t)e_ctx->g_capacity_frames * e_ctx->g_H4 * e_ctx->g_W4;
    e_ctx->g_mv_field = (MotionVector*)malloc(total * sizeof(MotionVector));

}

/* Ensure we have space for frame index f (0-based).
 * If f >= g_capacity_frames, grow the buffer with realloc.
 */
static void ensure_frame_capacity(int f)
{
    if (f < e_ctx->g_capacity_frames)
        return;

    int32_t new_cap = e_ctx->g_capacity_frames;
    while (f >= new_cap)
        new_cap *= 2;  // double until big enough

    size_t old_total = (size_t)e_ctx->g_capacity_frames * e_ctx->g_H4 * e_ctx->g_W4;
    size_t new_total = (size_t)new_cap * e_ctx->g_H4 * e_ctx->g_W4;

    e_ctx->g_mv_field = (MotionVector*)realloc(e_ctx->g_mv_field, new_total * sizeof(MotionVector));

//   (Optional) zero new area, in case there's residue
     memset(e_ctx->g_mv_field + old_total, 0, (new_total - old_total) * sizeof(MotionVector));

    e_ctx->g_capacity_frames = new_cap;
  
    printf("old tot = %zu, new tot = %zu MVs\n", old_total, new_total);
}



/* Accessor: MV for frame f at 4x4 coords (y4, x4).
 * If indices are out of range of what we've stored, return (0,0).
 */
static inline MotionVector *extractor_mv_at(int f, int y4, int x4)
{
    if (f  < 0 || y4 < 0 || x4 < 0)
        return &mv_zero;
    if (f  >= e_ctx->g_capacity_frames)
        return &mv_zero;
    if (y4 >= e_ctx->g_H4 || x4 >= e_ctx->g_W4)
        return &mv_zero;

    size_t idx = ((size_t)f * e_ctx->g_H4 + y4) * e_ctx->g_W4 + x4;
    return &e_ctx->g_mv_field[idx];
}


// the main extraction procedure
void extract_mvs(Macroblock *currMB)
{
  int list, by, bx;
  
  Slice *currSlice = currMB->p_Slice;
  VideoParameters *p_Vid     = currMB->p_Vid;
  StorablePicture *dec_pic   = currSlice->dec_picture;
  PicMotionParams **mv_info  = dec_pic->mv_info;
  int mbx_base = currMB->block_x;  // top-left 4x4 coord of MB (X)
  int mby_base = currMB->block_y;  // top-left 4x4 coord of MB (Y)
  int is_idr     = dec_pic->idr_flag;
  
  // if as not initialized, init
  if (!e_ctx) init_mv_extraction(p_Vid);

  // track idr frame-no resets
  if (is_idr && !e_ctx->entered_IDR_toggle) {
    e_ctx->entered_IDR_toggle = TRUE;
    e_ctx->frames_before_last_IDR = e_ctx->g_num_frames;
    ++e_ctx->IDR_cnt;
  }
  if (!is_idr) e_ctx->entered_IDR_toggle = FALSE;
  
  int f = dec_pic->frame_poc/2 + e_ctx->frames_before_last_IDR;  // current frame index (0-based) (use ThisPOC because frame-no does not refresh until later...
  
  ensure_frame_capacity(f); // Make sure we have space for this frame, dynamic array
  
  // Track the highest frame index we've seen so far.
//  fprintf(stdout, "f = %d\n", p_Vid->ThisPOC);
  if (f + 1 > e_ctx->g_num_frames)
      e_ctx->g_num_frames = f + 1;

  for (by = 0; by < BLOCK_MULTIPLE; ++by)
  {
    for (bx = 0; bx < BLOCK_MULTIPLE; ++bx)
    {
      for (list = 0; list < 2; ++list)
      /* Two lists in JM: 0 = LIST_0 (forward), 1 = LIST_1 (backward) */
      {
        if (list==LIST_0)  // only extract forward mvs for now
        {
          int y4 = mby_base + by;
          int x4 = mbx_base + bx;

          MotionVector *mv = &mv_info[y4][x4].mv[0];  // LIST_0

          MotionVector *slot = extractor_mv_at(f, y4, x4);
          slot->mv_x = mv->mv_x;
          slot->mv_y = mv->mv_y;
        }
      }
    }
  }
}


/* Write the MV field to a binary file after decoding is done.
 * We use g_num_frames as the actual number of frames seen.
 *
 * Format (little-endian), matches Python side:
 *   int32: num_frames (F)
 *   int32: H4
 *   int32: W4
 *   then F * H4 * W4 * (mv_x, mv_y) as int16 pairs (CustomMV)
 */
void write_mv_file(void)
{
    FILE *f = fopen(felix_mv_out_path, "wb");

    fwrite(&e_ctx->g_num_frames, sizeof(int32_t), 1, f);
    fwrite(&e_ctx->g_H4,         sizeof(int32_t), 1, f);
    fwrite(&e_ctx->g_W4,         sizeof(int32_t), 1, f);

    size_t total = (size_t)e_ctx->g_num_frames * e_ctx->g_H4 * e_ctx->g_W4;
    fwrite(e_ctx->g_mv_field, sizeof(MotionVector), total, f);
  
  
    // -------------------- DIAGNOSTICS START --------------------
    printf("\n============= EXTRACTION DIAGNOSTICS ================\n");
    printf("File pointer (before fclose): %p\n", (void*)f);
    printf("Total MotionVectors: %zu\n", total);

    int nonzero_count = 0;

    for (size_t i = 0; i < total; i++) {
        if (e_ctx->g_mv_field[i].mv_x != 0 || e_ctx->g_mv_field[i].mv_y != 0) {
            nonzero_count++;

            // Print first few non-zero MV entries
            if (nonzero_count <= 10) {
                printf("  Non-zero MV at index %zu: (mvx=%d, mvy=%d)\n",
                       i,
                       e_ctx->g_mv_field[i].mv_x,
                       e_ctx->g_mv_field[i].mv_y);
            }
        }
    }

    printf("Total non-zero MotionVectors: %d\n", nonzero_count);
    printf("=====================================================\n\n");
    // --------------------- DIAGNOSTICS END ---------------------
  
    fprintf(stdout,
            "total size of output %zu bytes\n",
            total * sizeof(MotionVector));

    fclose(f);
}


/* Cleanup if you want to free memory */
void free_mv_field(void)
{
    free(e_ctx->g_mv_field);
    e_ctx->g_mv_field      = NULL;
    e_ctx->g_num_frames    = 0;
    e_ctx->g_capacity_frames = 0;
    e_ctx->g_H4            = 0;
}


