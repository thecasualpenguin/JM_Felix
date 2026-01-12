"""
zero_motion_vectors.py - Set all motion vectors to zero for testing

This script:
1. Loads macroblock data from a binary file
2. Sets all motion vectors (mv_list0 and mv_list1) to zero
3. Writes the modified data to an output file
"""

import sys
import torch
from pathlib import Path
from mbdata_io import read_mbdata, write_mbdata, MBDataTensors


def zero_motion_vectors(input_path: str, output_path: str) -> None:
    """
    Read MB data, set all motion vectors to zero, and write to output.
    
    Args:
        input_path: Path to input .bin file
        output_path: Path to output .bin file
    """
    print(f"Reading MB data from: {input_path}")
    tensors = read_mbdata(input_path)
    
    # Get shape information
    header = tensors.header
    print(f"\nLoaded data:")
    print(f"  Frames: {header.num_frames}")
    print(f"  Frame size (MBs): {header.frame_height_mbs}x{header.frame_width_mbs}")
    print(f"  Total MBs: {header.total_mbs}")
    
    # Count non-zero motion vectors before zeroing
    nonzero_mv0 = (tensors.mv_list0 != 0).any(dim=-1).sum().item()
    nonzero_mv1 = (tensors.mv_list1 != 0).any(dim=-1).sum().item()
    print(f"\nMotion vectors before zeroing:")
    print(f"  Non-zero MVs in LIST_0: {nonzero_mv0}")
    print(f"  Non-zero MVs in LIST_1: {nonzero_mv1}")
    
    # Create zero tensors with the same shape and dtype
    zero_mv_list0 = torch.zeros_like(tensors.mv_list0)
    zero_mv_list1 = torch.zeros_like(tensors.mv_list1)
    
    # Create new MBDataTensors with zeroed motion vectors
    modified_tensors = MBDataTensors(
        header=tensors.header,
        mb_type=tensors.mb_type,
        cbp=tensors.cbp,
        qp=tensors.qp,
        is_intra=tensors.is_intra,
        b8mode=tensors.b8mode,
        b8pdir=tensors.b8pdir,
        i16mode=tensors.i16mode,
        c_ipred_mode=tensors.c_ipred_mode,
        transform_8x8=tensors.transform_8x8,
        mv_list0=zero_mv_list0,
        mv_list1=zero_mv_list1,
        ref_idx_list0=tensors.ref_idx_list0,
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=tensors.ipredmode,
        luma_cof=tensors.luma_cof,
        chroma_cof=tensors.chroma_cof,
    )
    
    # Verify all MVs are zero
    assert (modified_tensors.mv_list0 == 0).all(), "mv_list0 not all zero!"
    assert (modified_tensors.mv_list1 == 0).all(), "mv_list1 not all zero!"
    
    print(f"\nAll motion vectors have been set to zero.")
    
    # Write to output file
    print(f"\nWriting modified data to: {output_path}")
    write_mbdata(modified_tensors, output_path)
    
    print(f"\nDone! All motion vectors have been zeroed and saved to {output_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python zero_motion_vectors.py <input.bin> [output.bin]")
        print("\nIf output.bin is not provided, defaults to input_zeroed_mvs.bin")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Default output: add "_zeroed_mvs" before .bin extension
        input_file = Path(input_path)
        output_path = str(input_file.parent / f"{input_file.stem}_zeroed_mvs{input_file.suffix}")
    
    zero_motion_vectors(input_path, output_path)


if __name__ == '__main__':
    main()

