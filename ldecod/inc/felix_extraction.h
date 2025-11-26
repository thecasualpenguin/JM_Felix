
//
//  felix_extraction.h
//  ldecod
//
//  Created by Felix Gan on 11/24/25.
//

#ifndef felix_extraction_h
#define felix_extraction_h

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "global.h"
#include "mbuffer.h"
#include "felix.h"

static FelixVideoContext *e_ctx = NULL;   // context (global variables) for extractor

static MotionVector mv_zero = {0, 0};   // same with zero_mv but not const

void init_mv_extraction(VideoParameters *p_Vid);
static void ensure_frame_capacity(int f);
static inline MotionVector *extractor_mv_at(int f, int y4, int x4);
void extract_mvs(Macroblock *currMB);
void write_mv_file(void);
void free_mv_field(void);


#endif /* felix_extraction_h */
