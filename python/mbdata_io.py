"""
mbdata_io.py - Read and write JM decoder macroblock data files

This module provides functions to:
1. Load extracted MB data from binary files into PyTorch tensors
2. Save PyTorch tensors back to binary format for injection

Binary format (version 2.0):
    Header (32 bytes):
        magic_number      (uint32) - 0x464D4232 ("FMB2")
        version           (uint32) - 0x00020000
        num_frames        (int32)
        frame_height_mbs  (int32)
        frame_width_mbs   (int32)
        chroma_format_idc (int32)
        bytes_per_mb      (int32)
        reserved          (int32)

    Data:
        MacroblockData[num_frames * H_MB * W_MB]

Each MacroblockData (978 bytes for 4:2:0):
    - MBMetadata (16 bytes)
    - Block4x4Data[16] (12 bytes each = 192 bytes)
    - luma_cof[16][16] (512 bytes)
    - chroma_cof[2][64] (256 bytes)
"""

import struct
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from pathlib import Path


# Constants matching the C header
FELIX_MBDATA_MAGIC = 0x464D4232  # "FMB2"
FELIX_MBDATA_VERSION = 0x00020000

# Struct sizes (with packing)
HEADER_SIZE = 32
BLOCK4X4_SIZE = 12  # 8 (MVs) + 2 (ref_idx) + 1 (ipredmode) + 1 (reserved)
MBMETADATA_SIZE = 16
MBDATA_SIZE_420 = 978  # 16 + 192 + 512 + 256 + 2 padding


@dataclass
class MBDataHeader:
    """Header information from the binary file."""
    magic_number: int
    version: int
    num_frames: int
    frame_height_mbs: int
    frame_width_mbs: int
    chroma_format_idc: int
    bytes_per_mb: int
    reserved: int

    @property
    def total_mbs(self) -> int:
        return self.num_frames * self.frame_height_mbs * self.frame_width_mbs

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Returns (num_frames, H_MB, W_MB)."""
        return (self.num_frames, self.frame_height_mbs, self.frame_width_mbs)


@dataclass
class MBDataTensors:
    """PyTorch tensors containing all macroblock data."""

    # Header info
    header: MBDataHeader

    # MB Metadata - shape: (F, H_MB, W_MB)
    mb_type: torch.Tensor        # int16
    cbp: torch.Tensor            # int16
    qp: torch.Tensor             # int8
    is_intra: torch.Tensor       # bool
    b8mode: torch.Tensor         # int8, shape: (F, H_MB, W_MB, 4)
    b8pdir: torch.Tensor         # int8, shape: (F, H_MB, W_MB, 4)
    i16mode: torch.Tensor        # int8
    c_ipred_mode: torch.Tensor   # int8
    transform_8x8: torch.Tensor  # bool

    # Per 4x4 block data - shape: (F, H_MB, W_MB, 4, 4) for spatial, last dim for components
    mv_list0: torch.Tensor       # int16, shape: (F, H_MB, W_MB, 4, 4, 2) - (mv_x, mv_y)
    mv_list1: torch.Tensor       # int16, shape: (F, H_MB, W_MB, 4, 4, 2)
    ref_idx_list0: torch.Tensor  # int8, shape: (F, H_MB, W_MB, 4, 4)
    ref_idx_list1: torch.Tensor  # int8, shape: (F, H_MB, W_MB, 4, 4)
    ipredmode: torch.Tensor      # int8, shape: (F, H_MB, W_MB, 4, 4)

    # Coefficients
    luma_cof: torch.Tensor       # int16, shape: (F, H_MB, W_MB, 16, 16)
    chroma_cof: torch.Tensor     # int16, shape: (F, H_MB, W_MB, 2, 64)

    def to(self, device: torch.device) -> 'MBDataTensors':
        """Move all tensors to specified device."""
        return MBDataTensors(
            header=self.header,
            mb_type=self.mb_type.to(device),
            cbp=self.cbp.to(device),
            qp=self.qp.to(device),
            is_intra=self.is_intra.to(device),
            b8mode=self.b8mode.to(device),
            b8pdir=self.b8pdir.to(device),
            i16mode=self.i16mode.to(device),
            c_ipred_mode=self.c_ipred_mode.to(device),
            transform_8x8=self.transform_8x8.to(device),
            mv_list0=self.mv_list0.to(device),
            mv_list1=self.mv_list1.to(device),
            ref_idx_list0=self.ref_idx_list0.to(device),
            ref_idx_list1=self.ref_idx_list1.to(device),
            ipredmode=self.ipredmode.to(device),
            luma_cof=self.luma_cof.to(device),
            chroma_cof=self.chroma_cof.to(device),
        )

    def cuda(self) -> 'MBDataTensors':
        """Move all tensors to CUDA."""
        return self.to(torch.device('cuda'))

    def mps(self) -> 'MBDataTensors':
        """Move all tensors to MPS."""
        return self.to(torch.device('mps'))

    def cpu(self) -> 'MBDataTensors':
        """Move all tensors to CPU."""
        return self.to(torch.device('cpu'))


def read_mbdata(filepath: str) -> MBDataTensors:
    """
    Read macroblock data from binary file into PyTorch tensors.

    Args:
        filepath: Path to the .bin file created by EXTRACT_FULL_MODE

    Returns:
        MBDataTensors containing all MB data as PyTorch tensors
    """
    filepath = Path(filepath)

    with open(filepath, 'rb') as f:
        # Read header
        header_data = f.read(HEADER_SIZE)
        magic, version, num_frames, h_mb, w_mb, chroma_fmt, bytes_per_mb, reserved = \
            struct.unpack('<IIiiiiii', header_data)

        # Validate
        if magic != FELIX_MBDATA_MAGIC:
            raise ValueError(f"Invalid magic number: 0x{magic:08X} (expected 0x{FELIX_MBDATA_MAGIC:08X})")

        header = MBDataHeader(
            magic_number=magic,
            version=version,
            num_frames=num_frames,
            frame_height_mbs=h_mb,
            frame_width_mbs=w_mb,
            chroma_format_idc=chroma_fmt,
            bytes_per_mb=bytes_per_mb,
            reserved=reserved
        )

        print(f"Loading MB data: {num_frames} frames, {h_mb}x{w_mb} MBs per frame")
        print(f"Total MBs: {header.total_mbs}, File size: {bytes_per_mb * header.total_mbs + HEADER_SIZE} bytes")

        # Read all MB data as raw bytes
        raw_data = f.read()

    # Parse into numpy arrays first (faster), then convert to torch
    total_mbs = header.total_mbs

    # Define numpy dtype matching the C struct layout
    # MBMetadata (16 bytes)
    metadata_dtype = np.dtype([
        ('mb_type', '<i2'),
        ('cbp', '<i2'),
        ('qp', 'i1'),
        ('is_intra', 'i1'),
        ('b8mode', 'i1', (4,)),
        ('b8pdir', 'i1', (4,)),
        ('i16mode', 'i1'),
        ('c_ipred_mode', 'i1'),
        ('transform_8x8', 'i1'),
        ('reserved', 'i1'),
    ])

    # Block4x4Data (12 bytes each, 16 blocks)
    block_dtype = np.dtype([
        ('mv', '<i2', (2, 2)),      # mv[list][component] - 8 bytes
        ('ref_idx', 'i1', (2,)),    # ref_idx[list] - 2 bytes
        ('ipredmode', 'i1'),        # 1 byte
        ('reserved', 'i1'),         # 1 byte
    ])

    # Full MacroblockData
    mb_dtype = np.dtype([
        ('metadata', metadata_dtype),
        ('blocks', block_dtype, (16,)),
        ('luma_cof', '<i2', (16, 16)),
        ('chroma_cof', '<i2', (2, 64)),
    ])

    # Verify size
    expected_size = mb_dtype.itemsize
    if expected_size != bytes_per_mb:
        print(f"Warning: Expected {expected_size} bytes per MB, file says {bytes_per_mb}")

    # Parse raw data
    mb_array = np.frombuffer(raw_data, dtype=mb_dtype, count=total_mbs)

    # Reshape to (F, H_MB, W_MB)
    shape = header.shape
    mb_array = mb_array.reshape(shape)

    # Extract and convert to PyTorch tensors
    # Metadata
    mb_type = torch.from_numpy(mb_array['metadata']['mb_type'].copy())
    cbp = torch.from_numpy(mb_array['metadata']['cbp'].copy())
    qp = torch.from_numpy(mb_array['metadata']['qp'].copy())
    is_intra = torch.from_numpy(mb_array['metadata']['is_intra'].copy().astype(bool))
    b8mode = torch.from_numpy(mb_array['metadata']['b8mode'].copy())
    b8pdir = torch.from_numpy(mb_array['metadata']['b8pdir'].copy())
    i16mode = torch.from_numpy(mb_array['metadata']['i16mode'].copy())
    c_ipred_mode = torch.from_numpy(mb_array['metadata']['c_ipred_mode'].copy())
    transform_8x8 = torch.from_numpy(mb_array['metadata']['transform_8x8'].copy().astype(bool))

    # Block data - reshape from (F, H, W, 16) to (F, H, W, 4, 4)
    blocks = mb_array['blocks']  # shape: (F, H, W, 16)

    # MVs: blocks['mv'] shape is (F, H, W, 16, 2, 2) - [block][list][xy]
    mv_data = blocks['mv'].copy()  # (F, H, W, 16, 2, 2)
    mv_data = mv_data.reshape(*shape, 4, 4, 2, 2)  # (F, H, W, 4, 4, 2, 2)
    mv_list0 = torch.from_numpy(mv_data[..., 0, :].copy())  # (F, H, W, 4, 4, 2)
    mv_list1 = torch.from_numpy(mv_data[..., 1, :].copy())  # (F, H, W, 4, 4, 2)

    # Reference indices
    ref_data = blocks['ref_idx'].copy()  # (F, H, W, 16, 2)
    ref_data = ref_data.reshape(*shape, 4, 4, 2)
    ref_idx_list0 = torch.from_numpy(ref_data[..., 0].copy())
    ref_idx_list1 = torch.from_numpy(ref_data[..., 1].copy())

    # Intra prediction modes
    ipred_data = blocks['ipredmode'].copy()  # (F, H, W, 16)
    ipredmode = torch.from_numpy(ipred_data.reshape(*shape, 4, 4).copy())

    # Coefficients
    luma_cof = torch.from_numpy(mb_array['luma_cof'].copy())
    chroma_cof = torch.from_numpy(mb_array['chroma_cof'].copy())

    return MBDataTensors(
        header=header,
        mb_type=mb_type,
        cbp=cbp,
        qp=qp,
        is_intra=is_intra,
        b8mode=b8mode,
        b8pdir=b8pdir,
        i16mode=i16mode,
        c_ipred_mode=c_ipred_mode,
        transform_8x8=transform_8x8,
        mv_list0=mv_list0,
        mv_list1=mv_list1,
        ref_idx_list0=ref_idx_list0,
        ref_idx_list1=ref_idx_list1,
        ipredmode=ipredmode,
        luma_cof=luma_cof,
        chroma_cof=chroma_cof,
    )


def write_mbdata(tensors: MBDataTensors, filepath: str) -> None:
    """
    Write macroblock data from PyTorch tensors to binary file for injection.

    Args:
        tensors: MBDataTensors containing all MB data
        filepath: Output path for the .bin file
    """
    filepath = Path(filepath)
    header = tensors.header
    shape = header.shape
    total_mbs = header.total_mbs

    # Move to CPU and convert to numpy
    def to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    # Build numpy dtype matching C struct
    metadata_dtype = np.dtype([
        ('mb_type', '<i2'),
        ('cbp', '<i2'),
        ('qp', 'i1'),
        ('is_intra', 'i1'),
        ('b8mode', 'i1', (4,)),
        ('b8pdir', 'i1', (4,)),
        ('i16mode', 'i1'),
        ('c_ipred_mode', 'i1'),
        ('transform_8x8', 'i1'),
        ('reserved', 'i1'),
    ])

    block_dtype = np.dtype([
        ('mv', '<i2', (2, 2)),
        ('ref_idx', 'i1', (2,)),
        ('ipredmode', 'i1'),
        ('reserved', 'i1'),
    ])

    mb_dtype = np.dtype([
        ('metadata', metadata_dtype),
        ('blocks', block_dtype, (16,)),
        ('luma_cof', '<i2', (16, 16)),
        ('chroma_cof', '<i2', (2, 64)),
    ])

    # Create output array
    mb_array = np.zeros(shape, dtype=mb_dtype)

    # Fill metadata
    mb_array['metadata']['mb_type'] = to_numpy(tensors.mb_type).astype('<i2')
    mb_array['metadata']['cbp'] = to_numpy(tensors.cbp).astype('<i2')
    mb_array['metadata']['qp'] = to_numpy(tensors.qp).astype('i1')
    mb_array['metadata']['is_intra'] = to_numpy(tensors.is_intra).astype('i1')
    mb_array['metadata']['b8mode'] = to_numpy(tensors.b8mode).astype('i1')
    mb_array['metadata']['b8pdir'] = to_numpy(tensors.b8pdir).astype('i1')
    mb_array['metadata']['i16mode'] = to_numpy(tensors.i16mode).astype('i1')
    mb_array['metadata']['c_ipred_mode'] = to_numpy(tensors.c_ipred_mode).astype('i1')
    mb_array['metadata']['transform_8x8'] = to_numpy(tensors.transform_8x8).astype('i1')

    # Fill block data
    # Reshape MVs from (F, H, W, 4, 4, 2) back to (F, H, W, 16, 2, 2)
    mv_list0_np = to_numpy(tensors.mv_list0).reshape(*shape, 16, 2)  # (F, H, W, 16, 2)
    mv_list1_np = to_numpy(tensors.mv_list1).reshape(*shape, 16, 2)

    # Combine into (F, H, W, 16, 2, 2) - [block][list][xy]
    mv_combined = np.stack([mv_list0_np, mv_list1_np], axis=-2)  # (F, H, W, 16, 2, 2)
    mb_array['blocks']['mv'] = mv_combined.astype('<i2')

    # Reference indices
    ref0_np = to_numpy(tensors.ref_idx_list0).reshape(*shape, 16)
    ref1_np = to_numpy(tensors.ref_idx_list1).reshape(*shape, 16)
    ref_combined = np.stack([ref0_np, ref1_np], axis=-1)  # (F, H, W, 16, 2)
    mb_array['blocks']['ref_idx'] = ref_combined.astype('i1')

    # Intra prediction modes
    ipred_np = to_numpy(tensors.ipredmode).reshape(*shape, 16)
    mb_array['blocks']['ipredmode'] = ipred_np.astype('i1')

    # Coefficients
    mb_array['luma_cof'] = to_numpy(tensors.luma_cof).astype('<i2')
    mb_array['chroma_cof'] = to_numpy(tensors.chroma_cof).astype('<i2')

    # Write to file
    with open(filepath, 'wb') as f:
        # Write header
        header_bytes = struct.pack(
            '<IIiiiiii',
            FELIX_MBDATA_MAGIC,
            FELIX_MBDATA_VERSION,
            header.num_frames,
            header.frame_height_mbs,
            header.frame_width_mbs,
            header.chroma_format_idc,
            mb_dtype.itemsize,
            0  # reserved
        )
        f.write(header_bytes)

        # Write MB data
        f.write(mb_array.tobytes())

    print(f"Wrote {total_mbs} MBs to {filepath}")
    print(f"File size: {filepath.stat().st_size} bytes")


def get_tensor_summary(tensors: MBDataTensors) -> Dict:
    """Get summary statistics of the tensor data."""
    header = tensors.header

    return {
        'num_frames': header.num_frames,
        'frame_size_mbs': (header.frame_height_mbs, header.frame_width_mbs),
        'total_mbs': header.total_mbs,
        'intra_mbs': tensors.is_intra.sum().item(),
        'inter_mbs': (~tensors.is_intra).sum().item(),
        'nonzero_mv_list0': (tensors.mv_list0 != 0).any(dim=-1).sum().item(),
        'nonzero_mv_list1': (tensors.mv_list1 != 0).any(dim=-1).sum().item(),
        'nonzero_luma_cof': (tensors.luma_cof != 0).sum().item(),
        'nonzero_chroma_cof': (tensors.chroma_cof != 0).sum().item(),
        'qp_range': (tensors.qp.min().item(), tensors.qp.max().item()),
        'mv_list0_range': (tensors.mv_list0.min().item(), tensors.mv_list0.max().item()),
    }


# Example usage and testing
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mbdata_io.py <input.bin> [output.bin]")
        print("\nIf output.bin is provided, performs roundtrip test.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Read
    print(f"\n=== Reading {input_path} ===")
    tensors = read_mbdata(input_path)

    # Print summary
    print(f"\n=== Tensor Summary ===")
    summary = get_tensor_summary(tensors)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Print shapes
    print(f"\n=== Tensor Shapes ===")
    print(f"  mb_type: {tensors.mb_type.shape}")
    print(f"  mv_list0: {tensors.mv_list0.shape}")
    print(f"  mv_list1: {tensors.mv_list1.shape}")
    print(f"  luma_cof: {tensors.luma_cof.shape}")
    print(f"  chroma_cof: {tensors.chroma_cof.shape}")

    # Roundtrip test
    if output_path:
        print(f"\n=== Writing to {output_path} ===")
        write_mbdata(tensors, output_path)

        print(f"\n=== Verifying roundtrip ===")
        tensors2 = read_mbdata(output_path)

        # Compare
        all_match = True
        for field in ['mb_type', 'cbp', 'qp', 'is_intra', 'b8mode', 'b8pdir',
                      'i16mode', 'c_ipred_mode', 'transform_8x8',
                      'mv_list0', 'mv_list1', 'ref_idx_list0', 'ref_idx_list1',
                      'ipredmode', 'luma_cof', 'chroma_cof']:
            t1 = getattr(tensors, field)
            t2 = getattr(tensors2, field)
            if not torch.equal(t1, t2):
                print(f"  MISMATCH: {field}")
                all_match = False

        if all_match:
            print("  All tensors match!")
        else:
            print("  Some tensors differ!")
