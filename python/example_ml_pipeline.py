"""
example_ml_pipeline.py - Example ML pipeline using MB data

This shows how to use the mbdata_io module for machine learning tasks:
1. Load extracted MB data into PyTorch tensors
2. Process/modify the data (e.g., through a neural network)
3. Save modified data back for injection

Requirements:
    pip install numpy torch
"""

import torch
import torch.nn as nn
from mbdata_io import read_mbdata, write_mbdata, MBDataTensors, get_tensor_summary


class SimpleMVPredictor(nn.Module):
    """
    Example: Simple neural network that predicts motion vectors.
    This is just a placeholder - replace with your actual model.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Input: previous frame MVs (4x4x2 = 32 values per MB)
        # Output: current frame MVs (4x4x2 = 32 values per MB)
        self.net = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
        )

    def forward(self, prev_mvs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prev_mvs: (batch, 4, 4, 2) - MVs from previous frame
        Returns:
            predicted_mvs: (batch, 4, 4, 2) - predicted MVs
        """
        batch_size = prev_mvs.shape[0]
        x = prev_mvs.view(batch_size, -1).float()
        out = self.net(x)
        return out.view(batch_size, 4, 4, 2)


def example_modify_mvs(tensors: MBDataTensors) -> MBDataTensors:
    """
    Example: Modify motion vectors using a simple transformation.
    Replace this with your actual ML model inference.
    """
    # Clone to avoid modifying original
    new_mv_list0 = tensors.mv_list0.clone()
    new_mv_list1 = tensors.mv_list1.clone()

    # Example: Scale MVs by 0.9 (simple transformation)
    # In practice, you'd run your neural network here
    new_mv_list0 = (new_mv_list0.float() * 0.9).to(torch.int16)
    new_mv_list1 = (new_mv_list1.float() * 0.9).to(torch.int16)

    # Create new tensors object with modified MVs
    return MBDataTensors(
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
        mv_list0=new_mv_list0,
        mv_list1=new_mv_list1,
        ref_idx_list0=tensors.ref_idx_list0,
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=tensors.ipredmode,
        luma_cof=tensors.luma_cof,
        chroma_cof=tensors.chroma_cof,
    )


def prepare_mv_dataset(tensors: MBDataTensors) -> torch.utils.data.TensorDataset:
    """
    Prepare motion vector data for training.
    Creates pairs of (previous_frame_mvs, current_frame_mvs) for each MB.
    """
    F, H, W = tensors.header.shape

    # Flatten spatial dimensions
    mvs = tensors.mv_list0.view(F, H * W, 4, 4, 2)  # (F, H*W, 4, 4, 2)

    # Create input/output pairs (predict frame t from frame t-1)
    input_mvs = mvs[:-1].reshape(-1, 4, 4, 2)   # (F-1)*H*W samples
    target_mvs = mvs[1:].reshape(-1, 4, 4, 2)   # (F-1)*H*W samples

    return torch.utils.data.TensorDataset(input_mvs, target_mvs)


def prepare_coefficient_dataset(tensors: MBDataTensors) -> torch.utils.data.TensorDataset:
    """
    Prepare coefficient data for training.
    Returns (mb_metadata, luma_coefficients, chroma_coefficients) tuples.
    """
    F, H, W = tensors.header.shape
    total = F * H * W

    # Flatten and combine metadata into feature vector
    metadata = torch.stack([
        tensors.mb_type.view(total).float(),
        tensors.cbp.view(total).float(),
        tensors.qp.view(total).float(),
        tensors.is_intra.view(total).float(),
        tensors.i16mode.view(total).float(),
    ], dim=-1)  # (total, 5)

    luma = tensors.luma_cof.view(total, 16, 16)     # (total, 16, 16)
    chroma = tensors.chroma_cof.view(total, 2, 64)  # (total, 2, 64)

    return torch.utils.data.TensorDataset(metadata, luma, chroma)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python example_ml_pipeline.py <input.bin> [output.bin]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Load data
    print("Loading MB data...")
    tensors = read_mbdata(input_path)

    # Print summary
    print("\nData Summary:")
    summary = get_tensor_summary(tensors)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Example: Access specific tensors for ML
    print(f"\nTensor shapes:")
    print(f"  Motion Vectors (LIST_0): {tensors.mv_list0.shape}")
    print(f"    - Dimensions: (frames, MB_rows, MB_cols, block_row, block_col, xy)")
    print(f"  Luma Coefficients: {tensors.luma_cof.shape}")
    print(f"    - Dimensions: (frames, MB_rows, MB_cols, block_idx, coef_idx)")

    # Example: Get MVs for frame 10, MB at row 5 col 10
    if tensors.header.num_frames > 10:
        mv_sample = tensors.mv_list0[10, 5, 10]  # (4, 4, 2)
        print(f"\nSample MV (frame 10, MB [5,10]):")
        print(f"  Shape: {mv_sample.shape}")
        print(f"  Values:\n{mv_sample}")

    # Example: Create dataset for training
    print("\nCreating MV prediction dataset...")
    mv_dataset = prepare_mv_dataset(tensors)
    print(f"  Dataset size: {len(mv_dataset)} samples")
    print(f"  Sample input shape: {mv_dataset[0][0].shape}")
    print(f"  Sample target shape: {mv_dataset[0][1].shape}")

    # Example: Modify and save
    if output_path:
        print(f"\nModifying MVs and saving to {output_path}...")
        modified_tensors = example_modify_mvs(tensors)
        write_mbdata(modified_tensors, output_path)
        print("Done!")


if __name__ == '__main__':
    main()
