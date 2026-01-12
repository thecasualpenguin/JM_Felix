"""
mbdata_inspect.py - Inspect and visualize JM macroblock data dumps.

Usage:
  python mbdata_inspect.py /path/to/mbdata.bin --frame 0 --mb-row 0 --mb-col 0
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None

from mbdata_io import read_mbdata, get_tensor_summary


def _to_int(x):
    if hasattr(x, "item"):
        return x.item()
    return int(x)


def summarize_mb(tensors, frame_idx, mb_row, mb_col):
    """Return a JSON-serializable summary of a single macroblock."""
    summary = {
        "frame": frame_idx,
        "mb_row": mb_row,
        "mb_col": mb_col,
        "mb_type": _to_int(tensors.mb_type[frame_idx, mb_row, mb_col]),
        "cbp": _to_int(tensors.cbp[frame_idx, mb_row, mb_col]),
        "qp": _to_int(tensors.qp[frame_idx, mb_row, mb_col]),
        "is_intra": bool(tensors.is_intra[frame_idx, mb_row, mb_col].item()),
        "b8mode": tensors.b8mode[frame_idx, mb_row, mb_col].tolist(),
        "b8pdir": tensors.b8pdir[frame_idx, mb_row, mb_col].tolist(),
        "i16mode": _to_int(tensors.i16mode[frame_idx, mb_row, mb_col]),
        "c_ipred_mode": _to_int(tensors.c_ipred_mode[frame_idx, mb_row, mb_col]),
        "transform_8x8": bool(tensors.transform_8x8[frame_idx, mb_row, mb_col].item()),
        "mv_list0": tensors.mv_list0[frame_idx, mb_row, mb_col].tolist(),
        "mv_list1": tensors.mv_list1[frame_idx, mb_row, mb_col].tolist(),
        "ref_idx_list0": tensors.ref_idx_list0[frame_idx, mb_row, mb_col].tolist(),
        "ref_idx_list1": tensors.ref_idx_list1[frame_idx, mb_row, mb_col].tolist(),
        "ipredmode": tensors.ipredmode[frame_idx, mb_row, mb_col].tolist(),
        "luma_cof": tensors.luma_cof[frame_idx, mb_row, mb_col].tolist(),
        "chroma_cof": tensors.chroma_cof[frame_idx, mb_row, mb_col].tolist(),
    }
    return summary


def plot_frame_maps(tensors, frame_idx, outdir):
    """Save summary maps for a given frame."""
    mb_type = tensors.mb_type[frame_idx].cpu().numpy()
    qp = tensors.qp[frame_idx].cpu().numpy()
    is_intra = tensors.is_intra[frame_idx].cpu().numpy().astype(np.int32)
    cbp = tensors.cbp[frame_idx].cpu().numpy()

    mv = tensors.mv_list0[frame_idx].float().cpu().numpy()
    mv_mag = np.sqrt((mv[..., 0] ** 2 + mv[..., 1] ** 2))
    mv_mag = mv_mag.mean(axis=(2, 3))

    if HAS_MPL:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        axes = axes.flatten()

        for ax, data, title in [
            (axes[0], mb_type, "mb_type"),
            (axes[1], qp, "qp"),
            (axes[2], is_intra, "is_intra"),
            (axes[3], cbp, "cbp"),
            (axes[4], mv_mag, "mv_mag_list0"),
        ]:
            im = ax.imshow(data, interpolation="nearest")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[5].axis("off")
        out_path = outdir / f"frame_{frame_idx:04d}_maps.png"
        fig.suptitle(f"Frame {frame_idx} MB maps")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        write_ppm_grayscale(outdir / f"frame_{frame_idx:04d}_map_mb_type.ppm", mb_type)
        write_ppm_grayscale(outdir / f"frame_{frame_idx:04d}_map_qp.ppm", qp)
        write_ppm_grayscale(outdir / f"frame_{frame_idx:04d}_map_is_intra.ppm", is_intra)
        write_ppm_grayscale(outdir / f"frame_{frame_idx:04d}_map_cbp.ppm", cbp)
        write_ppm_grayscale(outdir / f"frame_{frame_idx:04d}_map_mv_mag_list0.ppm", mv_mag)


def plot_mv_quiver(tensors, frame_idx, outdir, stride=2):
    """Save a quiver plot of average MV per macroblock."""
    if not HAS_MPL:
        print("matplotlib not available: skipping MV quiver plot.")
        return

    mv = tensors.mv_list0[frame_idx].float().cpu().numpy()
    mv_avg = mv.mean(axis=(2, 3))

    H, W, _ = mv_avg.shape
    ys, xs = np.mgrid[0:H, 0:W]

    xs = xs[::stride, ::stride]
    ys = ys[::stride, ::stride]
    u = mv_avg[::stride, ::stride, 0]
    v = -mv_avg[::stride, ::stride, 1]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.quiver(xs, ys, u, v, angles="xy", scale_units="xy", scale=1.0)
    ax.set_aspect("equal")
    ax.set_title(f"Frame {frame_idx} MV list0 (avg per MB)")
    ax.invert_yaxis()

    out_path = outdir / f"frame_{frame_idx:04d}_mv_quiver.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coefficients(tensors, frame_idx, mb_row, mb_col, outdir):
    """Save coefficient heatmaps for a single macroblock."""
    luma = tensors.luma_cof[frame_idx, mb_row, mb_col].cpu().numpy()
    chroma = tensors.chroma_cof[frame_idx, mb_row, mb_col].cpu().numpy()
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        im0 = axes[0].imshow(luma, interpolation="nearest")
        axes[0].set_title("luma_cof (16x16)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(chroma, interpolation="nearest", aspect="auto")
        axes[1].set_title("chroma_cof (2x64)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        out_path = outdir / f"frame_{frame_idx:04d}_mb_{mb_row}_{mb_col}_coefs.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        write_ppm_grayscale(outdir / f"frame_{frame_idx:04d}_mb_{mb_row}_{mb_col}_luma.ppm", luma)
        write_ppm_grayscale(outdir / f"frame_{frame_idx:04d}_mb_{mb_row}_{mb_col}_chroma.ppm", chroma)


def write_ppm_grayscale(path, data):
    """Write a grayscale image as binary PPM (P6) without extra deps."""
    arr = np.array(data, dtype=np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v > min_v:
        norm = (arr - min_v) / (max_v - min_v)
    else:
        norm = np.zeros_like(arr)
    img = (norm * 255.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    h, w, _ = img.shape
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(img.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Inspect JM macroblock data dumps.")
    parser.add_argument("input_path", help="Path to .bin MB data file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to inspect")
    parser.add_argument("--mb-row", type=int, default=None, help="Macroblock row to inspect")
    parser.add_argument("--mb-col", type=int, default=None, help="Macroblock column to inspect")
    parser.add_argument("--outdir", default="python/mbdata_vis", help="Output directory for plots")
    parser.add_argument("--stride", type=int, default=2, help="Stride for MV quiver plot")
    args = parser.parse_args()

    tensors = read_mbdata(args.input_path)
    summary = get_tensor_summary(tensors)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    F, H, W = tensors.header.shape
    frame_idx = max(0, min(args.frame, F - 1))
    mb_row = args.mb_row if args.mb_row is not None else H // 2
    mb_col = args.mb_col if args.mb_col is not None else W // 2
    mb_row = max(0, min(mb_row, H - 1))
    mb_col = max(0, min(mb_col, W - 1))

    print("=== Header / Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Inspecting frame={frame_idx}, mb_row={mb_row}, mb_col={mb_col}")

    mb_summary = summarize_mb(tensors, frame_idx, mb_row, mb_col)
    summary_path = outdir / f"frame_{frame_idx:04d}_mb_{mb_row}_{mb_col}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(mb_summary, f, indent=2)

    print(f"Wrote MB summary: {summary_path}")

    plot_frame_maps(tensors, frame_idx, outdir)
    plot_mv_quiver(tensors, frame_idx, outdir, stride=max(1, args.stride))
    plot_coefficients(tensors, frame_idx, mb_row, mb_col, outdir)

    print(f"Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()
