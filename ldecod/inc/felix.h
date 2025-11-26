//
//  felix.h
//  ldecod
//
//  Created by Felix Gan on 11/17/25.
//

#ifndef felix_h
#define felix_h

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "global.h"
#include "mbuffer.h"

//static const char* mv_in_filepath = "/Users/havenville/mystuff/PhD/2025_09_26_Compressive_Generation/CompressiveGen/results/serialized_for_JM/premier_small_mv_1.bin";
static const char* mv_in_filepath = "/Users/havenville/mystuff/PhD/2025_09_26_Compressive_Generation/data/fgan/intermediate/mvs_extracted_from_JM.bin";

static const char* mv_log_path = "/Users/havenville/mystuff/PhD/2025_09_26_Compressive_Generation/data/fgan/intermediate/mvs.log";


typedef struct {
    int32_t g_num_frames;       // actual number of frames filled (max_frameno+1)
    int32_t g_capacity_frames;  // allocated frames
    int32_t g_H4;               // # of 4x4 rows
    int32_t g_W4;               // # of 4x4 cols
    MotionVector *g_mv_field;   // pointer to MV field buffer
    
    int32_t IDR_cnt;     // idr refreshes reset frame-no, so keep track
    int32_t frames_before_last_IDR;
    Boolean entered_IDR_toggle;    // prevents MB duplicate IDR incrementation
} FelixVideoContext;

static FelixVideoContext *i_ctx;   // injection context

FelixVideoContext *create_felix_context(void);
void print_mvs(Macroblock *currMB);

void get_custom_mv_for_block(const Macroblock *currMB, int list, int block_y, int block_x, MotionVector *out_mv);
void apply_custom_mvs_to_mb(Macroblock *currMB);
void apply_custom_mvs_to_mb_bare_minimum(Macroblock *currMB);

void sanity_check_block_multiple(void);
void load_mv_file(void);
static inline const MotionVector *mv_at(int f, int y4, int x4);


FILE *mv_log;
void open_mv_log(void);
void close_mv_log(void);





#endif /* felix_h */
