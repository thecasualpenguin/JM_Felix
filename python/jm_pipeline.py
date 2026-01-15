"""
jm_pipeline.py - Python interface for JM H.264 decoder extraction/injection pipeline

This module provides a clean Python API to:
1. Extract macroblock data from H.264 bitstreams using JM decoder
2. Read/modify the extracted data using PyTorch tensors
3. Inject modified data back during decoding
4. Re-encode decoded YUV to MP4 using FFmpeg

Example usage:
    from jm_pipeline import JMPipeline, MBDataProcessor

    # Create pipeline
    pipeline = JMPipeline(jm_path="/path/to/JM_modified_claude")

    # Define a processor (e.g., zero all motion vectors)
    def zero_mvs(tensors):
        tensors.mv_list0 = torch.zeros_like(tensors.mv_list0)
        tensors.mv_list1 = torch.zeros_like(tensors.mv_list1)
        return tensors

    # Run full pipeline
    output_mp4 = pipeline.process_video(
        input_h264="video.264",
        processor=zero_mvs,
        output_mp4="output.mp4"
    )
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from dataclasses import dataclass

# Import mbdata_io for tensor handling
from mbdata_io import read_mbdata, write_mbdata, MBDataTensors, get_tensor_summary


@dataclass
class VideoInfo:
    """Information about a decoded video."""
    width: int
    height: int
    num_frames: int
    chroma_format: str  # "420", "422", or "444"

    @property
    def frame_size_bytes(self) -> int:
        """Size of one frame in bytes (YUV420p assumed)."""
        if self.chroma_format == "420":
            return self.width * self.height * 3 // 2
        elif self.chroma_format == "422":
            return self.width * self.height * 2
        else:  # 444
            return self.width * self.height * 3


class JMPipeline:
    """
    Python interface for JM H.264 decoder extraction/injection pipeline.

    This class wraps the JM decoder to provide:
    - Extraction: Decode H.264 and extract full macroblock data
    - Injection: Decode H.264 with modified macroblock data
    - Integration with FFmpeg for final encoding
    """

    def __init__(
        self,
        jm_path: str = None,
        ffmpeg_path: str = "ffmpeg",
        verbose: bool = True
    ):
        """
        Initialize the JM pipeline.

        Args:
            jm_path: Path to JM_modified_claude directory. If None, uses parent of this file.
            ffmpeg_path: Path to ffmpeg executable (default: "ffmpeg" in PATH)
            verbose: Whether to print status messages
        """
        if jm_path is None:
            # Default to parent directory of this script
            jm_path = str(Path(__file__).parent.parent)

        self.jm_path = Path(jm_path)
        self.decoder_path = self.jm_path / "bin" / "ldecod.exe"
        self.ffmpeg_path = ffmpeg_path
        self.verbose = verbose

        # Verify decoder exists
        if not self.decoder_path.exists():
            raise FileNotFoundError(
                f"JM decoder not found at {self.decoder_path}. "
                f"Please build it with 'make -C {self.jm_path}/ldecod'"
            )

    def _log(self, msg: str):
        """Print a log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[JMPipeline] {msg}")

    def _is_mp4(self, filepath: str) -> bool:
        """Check if file is an MP4/MOV container (not raw H.264 Annex B)."""
        ext = Path(filepath).suffix.lower()
        return ext in ['.mp4', '.m4v', '.mov', '.mkv', '.avi', '.webm']

    def convert_to_annexb(self, input_file: str, output_file: str = None) -> str:
        """
        Convert MP4/container video to raw H.264 Annex B format for JM decoder.

        Args:
            input_file: Path to input video file (MP4, MOV, etc.)
            output_file: Path for output .264 file. If None, creates temp file.

        Returns:
            Path to the Annex B formatted .264 file
        """
        if output_file is None:
            fd, output_file = tempfile.mkstemp(suffix='.264', prefix='annexb_')
            os.close(fd)

        self._log(f"Converting {input_file} to Annex B format...")

        cmd = [
            self.ffmpeg_path,
            '-i', input_file,
            '-c:v', 'copy',
            '-bsf:v', 'h264_mp4toannexb',
            '-an',  # No audio
            '-y',   # Overwrite
            output_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Try without bitstream filter (might already be Annex B or different codec)
            cmd_simple = [
                self.ffmpeg_path,
                '-i', input_file,
                '-c:v', 'copy',
                '-an',
                '-y',
                output_file
            ]
            result = subprocess.run(cmd_simple, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg conversion failed:\n{result.stderr}\n"
                    f"Input file may not contain H.264 video."
                )

        self._log(f"Converted to: {output_file}")
        return output_file

    def _ensure_annexb(self, input_file: str, temp_dir: str = None) -> Tuple[str, bool]:
        """
        Ensure input is in Annex B format. Convert if necessary.

        Args:
            input_file: Input video file
            temp_dir: Directory for temp files

        Returns:
            Tuple of (annexb_path, was_converted)
        """
        if not self._is_mp4(input_file):
            # Assume already Annex B format
            return input_file, False

        # Need to convert
        if temp_dir:
            output = os.path.join(temp_dir, 'input.264')
        else:
            output = None

        annexb_path = self.convert_to_annexb(input_file, output)
        return annexb_path, True

    def _create_config(
        self,
        input_h264: str,
        output_yuv: str,
        mode: int,
        mbdata_input: str = None,
        mbdata_output: str = None,
    ) -> str:
        """
        Create a JM decoder config file.

        Args:
            input_h264: Path to input H.264 bitstream
            output_yuv: Path to output YUV file
            mode: Felix mode (0=extract_mv, 1=inject_mv, 2=extract_full, 3=inject_full)
            mbdata_input: Path to input mbdata file (for injection modes)
            mbdata_output: Path to output mbdata file (for extraction modes)

        Returns:
            Path to created config file
        """
        config_lines = [
            f'InputFile             = "{input_h264}"',
            f'OutputFile            = "{output_yuv}"',
            f'RefFile               = "test_rec.yuv"',
            f'',
            f'FelixCustomMode       = {mode}',
        ]

        if mbdata_input:
            config_lines.append(f'FelixMBDataInputFile  = "{mbdata_input}"')
        if mbdata_output:
            config_lines.append(f'FelixMBDataOutputFile = "{mbdata_output}"')

        config_lines.extend([
            '',
            'WriteUV               = 1',
            'FileFormat            = 0',
            'RefOffset             = 0',
            'POCScale              = 2',
            'DisplayDecParams      = 0',
            'ConcealMode           = 0',
            'RefPOCGap             = 2',
            'POCGap                = 2',
            'Silent                = 0',
            'IntraProfileDeblocking = 1',
            'DecFrmNum             = 0',
            'DecodeAllLayers       = 0',
        ])

        # Create temp config file
        config_fd, config_path = tempfile.mkstemp(suffix='.cfg', prefix='jm_')
        with os.fdopen(config_fd, 'w') as f:
            f.write('\n'.join(config_lines))

        return config_path

    def _run_decoder(self, config_path: str) -> subprocess.CompletedProcess:
        """Run the JM decoder with the given config."""
        cmd = [str(self.decoder_path), '-d', config_path]

        self._log(f"Running decoder: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.jm_path)
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"Decoder failed with return code {result.returncode}")

        return result

    def _parse_video_info(self, decoder_output: str) -> VideoInfo:
        """Parse video information from decoder output."""
        width = height = num_frames = None
        chroma_format = "420"

        for line in decoder_output.split('\n'):
            if 'Image Format' in line:
                # Parse "Image Format : 1280x720 (1280x720)"
                parts = line.split(':')[1].strip().split()[0]
                w, h = parts.split('x')
                width, height = int(w), int(h)
            elif 'frames are decoded' in line:
                # Parse "253 frames are decoded."
                num_frames = int(line.split()[0])
            elif 'Color Format' in line:
                if '4:2:0' in line:
                    chroma_format = "420"
                elif '4:2:2' in line:
                    chroma_format = "422"
                elif '4:4:4' in line:
                    chroma_format = "444"

        if width is None or height is None:
            raise RuntimeError("Could not parse video dimensions from decoder output")
        if num_frames is None:
            raise RuntimeError("Could not parse frame count from decoder output")

        return VideoInfo(
            width=width,
            height=height,
            num_frames=num_frames,
            chroma_format=chroma_format
        )

    def extract(
        self,
        input_h264: str,
        output_mbdata: str = None,
        output_yuv: str = None,
        keep_yuv: bool = False
    ) -> Tuple[MBDataTensors, VideoInfo, Optional[str]]:
        """
        Extract macroblock data from an H.264 bitstream.

        Args:
            input_h264: Path to input H.264 bitstream
            output_mbdata: Path to output mbdata file (optional, uses temp if not provided)
            output_yuv: Path to output YUV file (optional, uses temp if not provided)
            keep_yuv: Whether to keep the YUV file after extraction

        Returns:
            Tuple of (MBDataTensors, VideoInfo, yuv_path or None)
        """
        input_file = str(Path(input_h264).resolve())

        # Convert MP4 to Annex B if necessary
        input_h264, was_converted = self._ensure_annexb(input_file)
        cleanup_annexb = was_converted

        # Create temp files if needed
        if output_mbdata is None:
            mbdata_fd, output_mbdata = tempfile.mkstemp(suffix='.bin', prefix='mbdata_')
            os.close(mbdata_fd)
            cleanup_mbdata = True
        else:
            output_mbdata = str(Path(output_mbdata).resolve())
            cleanup_mbdata = False

        if output_yuv is None:
            yuv_fd, output_yuv = tempfile.mkstemp(suffix='.yuv', prefix='decoded_')
            os.close(yuv_fd)
            cleanup_yuv = not keep_yuv
        else:
            output_yuv = str(Path(output_yuv).resolve())
            cleanup_yuv = False

        try:
            # Create config for extraction (mode 2)
            config_path = self._create_config(
                input_h264=input_h264,
                output_yuv=output_yuv,
                mode=2,  # EXTRACT_FULL_MODE
                mbdata_output=output_mbdata
            )

            self._log(f"Extracting from {input_h264}")

            # Run decoder
            result = self._run_decoder(config_path)

            # Parse video info
            video_info = self._parse_video_info(result.stdout)
            self._log(f"Decoded {video_info.num_frames} frames at {video_info.width}x{video_info.height}")

            # Read mbdata
            self._log(f"Reading mbdata from {output_mbdata}")
            tensors = read_mbdata(output_mbdata)

            # Cleanup config
            os.unlink(config_path)

            # Cleanup temp mbdata if we created it
            if cleanup_mbdata:
                os.unlink(output_mbdata)

            # Cleanup converted annexb file if we created it
            if cleanup_annexb and os.path.exists(input_h264):
                os.unlink(input_h264)

            # Return YUV path if keeping, otherwise cleanup
            if cleanup_yuv:
                os.unlink(output_yuv)
                return tensors, video_info, None
            else:
                return tensors, video_info, output_yuv

        except Exception as e:
            # Cleanup on error
            if cleanup_mbdata and os.path.exists(output_mbdata):
                os.unlink(output_mbdata)
            if cleanup_yuv and os.path.exists(output_yuv):
                os.unlink(output_yuv)
            if cleanup_annexb and os.path.exists(input_h264):
                os.unlink(input_h264)
            raise

    def inject(
        self,
        input_h264: str,
        mbdata: Union[str, MBDataTensors],
        output_yuv: str = None,
    ) -> Tuple[str, VideoInfo]:
        """
        Decode H.264 with injected macroblock data.

        Args:
            input_h264: Path to input H.264 bitstream
            mbdata: Either path to mbdata file or MBDataTensors object
            output_yuv: Path to output YUV file (optional, uses temp if not provided)

        Returns:
            Tuple of (yuv_path, VideoInfo)
        """
        input_file = str(Path(input_h264).resolve())

        # Convert MP4 to Annex B if necessary
        input_h264, was_converted = self._ensure_annexb(input_file)
        cleanup_annexb = was_converted

        # Handle mbdata input
        if isinstance(mbdata, MBDataTensors):
            # Write tensors to temp file
            mbdata_fd, mbdata_path = tempfile.mkstemp(suffix='.bin', prefix='mbdata_inject_')
            os.close(mbdata_fd)
            write_mbdata(mbdata, mbdata_path)
            cleanup_mbdata = True
        else:
            mbdata_path = str(Path(mbdata).resolve())
            cleanup_mbdata = False

        # Create output YUV path
        if output_yuv is None:
            yuv_fd, output_yuv = tempfile.mkstemp(suffix='.yuv', prefix='injected_')
            os.close(yuv_fd)
        else:
            output_yuv = str(Path(output_yuv).resolve())

        try:
            # Create config for injection (mode 3)
            config_path = self._create_config(
                input_h264=input_h264,
                output_yuv=output_yuv,
                mode=3,  # INJECT_FULL_MODE
                mbdata_input=mbdata_path
            )

            self._log(f"Injecting {mbdata} into {input_h264}")

            # Run decoder
            result = self._run_decoder(config_path)

            # Parse video info
            video_info = self._parse_video_info(result.stdout)
            self._log(f"Decoded {video_info.num_frames} frames at {video_info.width}x{video_info.height}")

            # Cleanup
            os.unlink(config_path)
            if cleanup_mbdata:
                os.unlink(mbdata_path)
            if cleanup_annexb and os.path.exists(input_h264):
                os.unlink(input_h264)

            return output_yuv, video_info

        except Exception as e:
            if cleanup_mbdata and os.path.exists(mbdata_path):
                os.unlink(mbdata_path)
            if cleanup_annexb and os.path.exists(input_h264):
                os.unlink(input_h264)
            raise

    def yuv_to_mp4(
        self,
        input_yuv: str,
        output_mp4: str,
        video_info: VideoInfo,
        crf: int = 18,
        preset: str = "medium",
        extra_args: list = None
    ) -> str:
        """
        Convert YUV to MP4 using FFmpeg.

        Args:
            input_yuv: Path to input YUV file
            output_mp4: Path to output MP4 file
            video_info: VideoInfo with dimensions and format
            crf: x264 CRF value (0-51, lower = better quality)
            preset: x264 preset (ultrafast, fast, medium, slow, etc.)
            extra_args: Additional FFmpeg arguments

        Returns:
            Path to output MP4 file
        """
        # Determine pixel format
        if video_info.chroma_format == "420":
            pix_fmt = "yuv420p"
        elif video_info.chroma_format == "422":
            pix_fmt = "yuv422p"
        else:
            pix_fmt = "yuv444p"

        cmd = [
            self.ffmpeg_path,
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-pix_fmt', pix_fmt,
            '-s', f'{video_info.width}x{video_info.height}',
            '-i', input_yuv,
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', preset,
            '-pix_fmt', 'yuv420p',  # Output format
        ]

        if extra_args:
            cmd.extend(extra_args)

        cmd.append(output_mp4)

        self._log(f"Encoding to MP4: {output_mp4}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg STDERR:\n{result.stderr}")
            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")

        return output_mp4

    def process_video(
        self,
        input_h264: str,
        processor: Callable[[MBDataTensors], MBDataTensors],
        output_mp4: str,
        keep_intermediates: bool = False,
        intermediate_dir: str = None,
        crf: int = 18
    ) -> str:
        """
        Full pipeline: extract -> process -> inject -> encode.

        Args:
            input_h264: Path to input H.264 bitstream (if not, will convert to Annex B)
            processor: Function that takes MBDataTensors and returns modified MBDataTensors
            output_mp4: Path to output MP4 file
            keep_intermediates: Whether to keep intermediate files
            intermediate_dir: Directory for intermediate files (uses temp if not provided)
            crf: x264 CRF value for output encoding

        Returns:
            Path to output MP4 file
        """
        input_file = str(Path(input_h264).resolve())
        output_mp4 = str(Path(output_mp4).resolve())

        # Setup intermediate directory
        if intermediate_dir:
            intermediate_dir = Path(intermediate_dir)
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            cleanup_dir = False
        else:
            intermediate_dir = Path(tempfile.mkdtemp(prefix='jm_pipeline_'))
            cleanup_dir = not keep_intermediates

        # Convert MP4 to Annex B once (reused for extract and inject)
        input_h264, was_converted = self._ensure_annexb(input_file, str(intermediate_dir))

        try:
            # Step 1: Extract
            self._log("Step 1: Extracting macroblock data...")
            mbdata_extracted = intermediate_dir / "mbdata_extracted.bin"
            yuv_extracted = intermediate_dir / "extracted.yuv"

            tensors, video_info, _ = self.extract(
                input_h264=input_h264,  # Already annexb, won't convert again
                output_mbdata=str(mbdata_extracted),
                output_yuv=str(yuv_extracted),
                keep_yuv=keep_intermediates
            )

            # Print summary
            summary = get_tensor_summary(tensors)
            self._log(f"Extracted: {summary['num_frames']} frames, "
                     f"{summary['total_mbs']} MBs, "
                     f"{summary['inter_mbs']} inter, {summary['intra_mbs']} intra")

            # Step 2: Process
            self._log("Step 2: Processing macroblock data...")
            processed_tensors = processor(tensors)

            # Step 3: Inject
            self._log("Step 3: Injecting processed data...")
            mbdata_processed = intermediate_dir / "mbdata_processed.bin"
            yuv_injected = intermediate_dir / "injected.yuv"

            write_mbdata(processed_tensors, str(mbdata_processed))

            yuv_path, _ = self.inject(
                input_h264=input_h264,
                mbdata=str(mbdata_processed),
                output_yuv=str(yuv_injected)
            )

            # Step 4: Encode to MP4
            self._log("Step 4: Encoding to MP4...")
            self.yuv_to_mp4(
                input_yuv=yuv_path,
                output_mp4=output_mp4,
                video_info=video_info,
                crf=crf
            )

            self._log(f"Done! Output saved to {output_mp4}")

            return output_mp4

        finally:
            if cleanup_dir and intermediate_dir.exists():
                shutil.rmtree(intermediate_dir)


# ==================== Example Processors ====================

def zero_all_motion_vectors(tensors: MBDataTensors) -> MBDataTensors:
    """Set all motion vectors to zero."""
    import torch

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
        mv_list0=torch.zeros_like(tensors.mv_list0),
        mv_list1=torch.zeros_like(tensors.mv_list1),
        ref_idx_list0=tensors.ref_idx_list0,
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=tensors.ipredmode,
        luma_cof=tensors.luma_cof,
        chroma_cof=tensors.chroma_cof,
    )


def scale_motion_vectors(scale: float) -> Callable[[MBDataTensors], MBDataTensors]:
    """Create a processor that scales motion vectors by a factor."""
    def processor(tensors: MBDataTensors) -> MBDataTensors:
        import torch

        scaled_mv0 = (tensors.mv_list0.float() * scale).to(tensors.mv_list0.dtype)
        scaled_mv1 = (tensors.mv_list1.float() * scale).to(tensors.mv_list1.dtype)

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
            mv_list0=scaled_mv0,
            mv_list1=scaled_mv1,
            ref_idx_list0=tensors.ref_idx_list0,
            ref_idx_list1=tensors.ref_idx_list1,
            ipredmode=tensors.ipredmode,
            luma_cof=tensors.luma_cof,
            chroma_cof=tensors.chroma_cof,
        )
    return processor


def identity_processor(tensors: MBDataTensors) -> MBDataTensors:
    """Pass through without modification (for testing roundtrip)."""
    return tensors


# ==================== Residual Processors ====================

def zero_residuals(tensors: MBDataTensors) -> MBDataTensors:
    """Zero all residual coefficients (luma and chroma).

    This removes all residual detail, leaving only prediction.
    The result will look blocky/blurry as fine detail is lost.
    """
    import torch

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
        mv_list0=tensors.mv_list0,
        mv_list1=tensors.mv_list1,
        ref_idx_list0=tensors.ref_idx_list0,
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=tensors.ipredmode,
        luma_cof=torch.zeros_like(tensors.luma_cof),
        chroma_cof=torch.zeros_like(tensors.chroma_cof),
    )


def zero_luma_residuals(tensors: MBDataTensors) -> MBDataTensors:
    """Zero only luma residuals, keeping chroma intact.

    This removes brightness detail while preserving color information.
    """
    import torch

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
        mv_list0=tensors.mv_list0,
        mv_list1=tensors.mv_list1,
        ref_idx_list0=tensors.ref_idx_list0,
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=tensors.ipredmode,
        luma_cof=torch.zeros_like(tensors.luma_cof),
        chroma_cof=tensors.chroma_cof,
    )


def zero_chroma_residuals(tensors: MBDataTensors) -> MBDataTensors:
    """Zero only chroma residuals, keeping luma intact.

    This removes color detail while preserving brightness.
    Colors will be flat/posterized based on prediction.
    """
    import torch

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
        mv_list0=tensors.mv_list0,
        mv_list1=tensors.mv_list1,
        ref_idx_list0=tensors.ref_idx_list0,
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=tensors.ipredmode,
        luma_cof=tensors.luma_cof,
        chroma_cof=torch.zeros_like(tensors.chroma_cof),
    )


def scale_residuals(scale: float) -> Callable[[MBDataTensors], MBDataTensors]:
    """Create a processor that scales residual coefficients.

    Args:
        scale: Factor to multiply residuals by.
               < 1.0: Reduces detail (smoother, blurrier)
               > 1.0: Amplifies detail (sharper, potentially noisy/artifacts)
               Negative: Inverts residuals (trippy effect!)
    """
    def processor(tensors: MBDataTensors) -> MBDataTensors:
        import torch

        scaled_luma = (tensors.luma_cof.float() * scale).to(tensors.luma_cof.dtype)
        scaled_chroma = (tensors.chroma_cof.float() * scale).to(tensors.chroma_cof.dtype)

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
            mv_list0=tensors.mv_list0,
            mv_list1=tensors.mv_list1,
            ref_idx_list0=tensors.ref_idx_list0,
            ref_idx_list1=tensors.ref_idx_list1,
            ipredmode=tensors.ipredmode,
            luma_cof=scaled_luma,
            chroma_cof=scaled_chroma,
        )
    return processor


def quantize_residuals(levels: int = 8) -> Callable[[MBDataTensors], MBDataTensors]:
    """Create a processor that quantizes residuals to fewer discrete levels.

    This creates a posterization/banding effect by reducing precision.

    Args:
        levels: Number of quantization levels. Lower = more extreme effect.
                8 = subtle banding, 2 = very harsh/artistic
    """
    def processor(tensors: MBDataTensors) -> MBDataTensors:
        import torch

        def quantize(tensor, num_levels):
            t_float = tensor.float()
            t_min, t_max = t_float.min(), t_float.max()
            if t_max == t_min:
                return tensor
            normalized = (t_float - t_min) / (t_max - t_min)
            quantized = torch.round(normalized * (num_levels - 1)) / (num_levels - 1)
            return ((quantized * (t_max - t_min)) + t_min).to(tensor.dtype)

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
            mv_list0=tensors.mv_list0,
            mv_list1=tensors.mv_list1,
            ref_idx_list0=tensors.ref_idx_list0,
            ref_idx_list1=tensors.ref_idx_list1,
            ipredmode=tensors.ipredmode,
            luma_cof=quantize(tensors.luma_cof, levels),
            chroma_cof=quantize(tensors.chroma_cof, levels),
        )
    return processor


def add_residual_noise(stddev: float = 10.0) -> Callable[[MBDataTensors], MBDataTensors]:
    """Create a processor that adds Gaussian noise to residuals.

    This creates a grainy/noisy film effect.

    Args:
        stddev: Standard deviation of the noise. Higher = more noisy.
    """
    def processor(tensors: MBDataTensors) -> MBDataTensors:
        import torch

        luma_noise = torch.randn_like(tensors.luma_cof.float()) * stddev
        chroma_noise = torch.randn_like(tensors.chroma_cof.float()) * stddev

        noisy_luma = (tensors.luma_cof.float() + luma_noise).to(tensors.luma_cof.dtype)
        noisy_chroma = (tensors.chroma_cof.float() + chroma_noise).to(tensors.chroma_cof.dtype)

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
            mv_list0=tensors.mv_list0,
            mv_list1=tensors.mv_list1,
            ref_idx_list0=tensors.ref_idx_list0,
            ref_idx_list1=tensors.ref_idx_list1,
            ipredmode=tensors.ipredmode,
            luma_cof=noisy_luma,
            chroma_cof=noisy_chroma,
        )
    return processor


# ==================== Combined & Metadata Processors ====================

def zero_all(tensors: MBDataTensors) -> MBDataTensors:
    """Zero motion vectors and all residual coefficients.

    Sanity check: zeros MVs and residuals while keeping all metadata intact.
    Inter blocks will copy from reference at same position (zero motion).
    Intra blocks will use prediction only with no residual correction.

    Should produce a very corrupted/blocky video with no detail.
    """
    import torch

    return MBDataTensors(
        header=tensors.header,
        mb_type=tensors.mb_type,
        cbp=tensors.cbp,  # Keep original CBP
        qp=tensors.qp,
        is_intra=tensors.is_intra,
        b8mode=tensors.b8mode,  # Keep original partition modes
        b8pdir=tensors.b8pdir,  # Keep original prediction directions
        i16mode=tensors.i16mode,  # Keep original I16 mode
        c_ipred_mode=tensors.c_ipred_mode,  # Keep original chroma mode
        transform_8x8=tensors.transform_8x8,
        mv_list0=torch.zeros_like(tensors.mv_list0),  # Zero motion vectors
        mv_list1=torch.zeros_like(tensors.mv_list1),
        ref_idx_list0=tensors.ref_idx_list0,  # Keep original ref indices
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=tensors.ipredmode,  # Keep original intra modes
        luma_cof=torch.zeros_like(tensors.luma_cof),  # Zero luma residuals
        chroma_cof=torch.zeros_like(tensors.chroma_cof),  # Zero chroma residuals
    )


def invert_motion_vectors(tensors: MBDataTensors) -> MBDataTensors:
    """Negate all motion vectors (multiply by -1).

    Creates a "reverse motion" effect - objects move opposite to their
    intended direction, causing ghosting and trailing artifacts.
    """
    import torch

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
        mv_list0=-tensors.mv_list0,
        mv_list1=-tensors.mv_list1,
        ref_idx_list0=tensors.ref_idx_list0,
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=tensors.ipredmode,
        luma_cof=tensors.luma_cof,
        chroma_cof=tensors.chroma_cof,
    )


def randomize_intra_modes(tensors: MBDataTensors) -> MBDataTensors:
    """Randomize intra prediction modes (0-8 for 4x4 luma).

    Creates blocky, glitchy artifacts on intra-coded blocks as
    the wrong prediction directions are used.
    """
    import torch

    # Intra 4x4 modes are 0-8 (9 modes: DC + 8 directional)
    random_modes = torch.randint(0, 9, tensors.ipredmode.shape, dtype=tensors.ipredmode.dtype)

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
        mv_list0=tensors.mv_list0,
        mv_list1=tensors.mv_list1,
        ref_idx_list0=tensors.ref_idx_list0,
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=random_modes,
        luma_cof=tensors.luma_cof,
        chroma_cof=tensors.chroma_cof,
    )


def shift_qp(delta: int = 10) -> Callable[[MBDataTensors], MBDataTensors]:
    """Create a processor that shifts QP values.

    Args:
        delta: Amount to add to QP. Positive = coarser quantization (blockier),
               Negative = finer quantization. Valid QP range is 0-51.
    """
    def processor(tensors: MBDataTensors) -> MBDataTensors:
        import torch

        # Shift and clamp to valid range
        shifted_qp = torch.clamp(tensors.qp.int() + delta, 0, 51).to(tensors.qp.dtype)

        return MBDataTensors(
            header=tensors.header,
            mb_type=tensors.mb_type,
            cbp=tensors.cbp,
            qp=shifted_qp,
            is_intra=tensors.is_intra,
            b8mode=tensors.b8mode,
            b8pdir=tensors.b8pdir,
            i16mode=tensors.i16mode,
            c_ipred_mode=tensors.c_ipred_mode,
            transform_8x8=tensors.transform_8x8,
            mv_list0=tensors.mv_list0,
            mv_list1=tensors.mv_list1,
            ref_idx_list0=tensors.ref_idx_list0,
            ref_idx_list1=tensors.ref_idx_list1,
            ipredmode=tensors.ipredmode,
            luma_cof=tensors.luma_cof,
            chroma_cof=tensors.chroma_cof,
        )
    return processor


def swap_chroma_channels(tensors: MBDataTensors) -> MBDataTensors:
    """Swap U and V chroma channels.

    Creates color shift effects - reds become blues, etc.
    """
    import torch

    # chroma_cof shape is [frames, mbs, 2, 64] where dim 2 is [Cb, Cr]
    swapped_chroma = tensors.chroma_cof.clone()
    swapped_chroma[:, :, 0, :] = tensors.chroma_cof[:, :, 1, :]
    swapped_chroma[:, :, 1, :] = tensors.chroma_cof[:, :, 0, :]

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
        mv_list0=tensors.mv_list0,
        mv_list1=tensors.mv_list1,
        ref_idx_list0=tensors.ref_idx_list0,
        ref_idx_list1=tensors.ref_idx_list1,
        ipredmode=tensors.ipredmode,
        luma_cof=tensors.luma_cof,
        chroma_cof=swapped_chroma,
    )


def amplify_motion(factor: float = 2.0) -> Callable[[MBDataTensors], MBDataTensors]:
    """Create a processor that amplifies motion vectors.

    Args:
        factor: Multiplication factor. >1 = exaggerated motion/blur,
                <1 = reduced motion, 0 = freeze frame effect
    """
    def processor(tensors: MBDataTensors) -> MBDataTensors:
        import torch

        amp_mv0 = (tensors.mv_list0.float() * factor).to(tensors.mv_list0.dtype)
        amp_mv1 = (tensors.mv_list1.float() * factor).to(tensors.mv_list1.dtype)

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
            mv_list0=amp_mv0,
            mv_list1=amp_mv1,
            ref_idx_list0=tensors.ref_idx_list0,
            ref_idx_list1=tensors.ref_idx_list1,
            ipredmode=tensors.ipredmode,
            luma_cof=tensors.luma_cof,
            chroma_cof=tensors.chroma_cof,
        )
    return processor


# ==================== CLI Interface ====================

def main():
    """Command-line interface for the JM pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="JM H.264 Decoder Pipeline - Extract, Process, Inject, Encode"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract macroblock data from H.264')
    extract_parser.add_argument('input', help='Input H.264 bitstream')
    extract_parser.add_argument('-o', '--output', help='Output mbdata file')
    extract_parser.add_argument('--yuv', help='Output YUV file')
    extract_parser.add_argument('--jm-path', help='Path to JM_modified_claude directory')

    # Inject command
    inject_parser = subparsers.add_parser('inject', help='Inject macroblock data during decode')
    inject_parser.add_argument('input', help='Input H.264 bitstream')
    inject_parser.add_argument('mbdata', help='Input mbdata file')
    inject_parser.add_argument('-o', '--output', help='Output YUV file')
    inject_parser.add_argument('--mp4', help='Also encode to MP4')
    inject_parser.add_argument('--jm-path', help='Path to JM_modified_claude directory')

    # Process command
    process_parser = subparsers.add_parser('process', help='Full pipeline: extract, process, inject, encode')
    process_parser.add_argument('input', help='Input H.264 bitstream')
    process_parser.add_argument('output', help='Output MP4 file')
    process_parser.add_argument('--processor', choices=[
        'identity', 'zero_mvs', 'scale_mvs', 'invert_mvs', 'amplify_mvs',
        'zero_residuals', 'zero_luma', 'zero_chroma',
        'scale_residuals', 'quantize_residuals', 'noise_residuals',
        'zero_all', 'randomize_intra', 'shift_qp', 'swap_chroma'
    ], default='identity', help='Processing to apply')
    process_parser.add_argument('--scale', type=float, default=0.5, help='Scale factor for scale_mvs/scale_residuals')
    process_parser.add_argument('--amplify', type=float, default=2.0, help='Amplification factor for amplify_mvs')
    process_parser.add_argument('--levels', type=int, default=8, help='Quantization levels for quantize_residuals')
    process_parser.add_argument('--noise-stddev', type=float, default=10.0, help='Noise stddev for noise_residuals')
    process_parser.add_argument('--qp-delta', type=int, default=10, help='QP shift amount for shift_qp')
    process_parser.add_argument('--crf', type=int, default=18, help='x264 CRF value')
    process_parser.add_argument('--keep', action='store_true', help='Keep intermediate files')
    process_parser.add_argument('--intermediate-dir', help='Directory for intermediate files')
    process_parser.add_argument('--jm-path', help='Path to JM_modified_claude directory')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Create pipeline
    jm_path = getattr(args, 'jm_path', None)
    pipeline = JMPipeline(jm_path=jm_path)

    if args.command == 'extract':
        tensors, video_info, yuv_path = pipeline.extract(
            input_h264=args.input,
            output_mbdata=args.output,
            output_yuv=args.yuv,
            keep_yuv=args.yuv is not None
        )
        summary = get_tensor_summary(tensors)
        print(f"\nExtraction complete!")
        print(f"  Video: {video_info.width}x{video_info.height}, {video_info.num_frames} frames")
        print(f"  MBs: {summary['total_mbs']} total, {summary['inter_mbs']} inter, {summary['intra_mbs']} intra")
        if args.output:
            print(f"  MBData saved to: {args.output}")
        if yuv_path:
            print(f"  YUV saved to: {yuv_path}")

    elif args.command == 'inject':
        yuv_path, video_info = pipeline.inject(
            input_h264=args.input,
            mbdata=args.mbdata,
            output_yuv=args.output
        )
        print(f"\nInjection complete!")
        print(f"  Video: {video_info.width}x{video_info.height}, {video_info.num_frames} frames")
        print(f"  YUV saved to: {yuv_path}")

        if args.mp4:
            pipeline.yuv_to_mp4(yuv_path, args.mp4, video_info)
            print(f"  MP4 saved to: {args.mp4}")

    elif args.command == 'process':
        # Select processor
        if args.processor == 'identity':
            processor = identity_processor
        elif args.processor == 'zero_mvs':
            processor = zero_all_motion_vectors
        elif args.processor == 'scale_mvs':
            processor = scale_motion_vectors(args.scale)
        elif args.processor == 'invert_mvs':
            processor = invert_motion_vectors
        elif args.processor == 'amplify_mvs':
            processor = amplify_motion(args.amplify)
        elif args.processor == 'zero_residuals':
            processor = zero_residuals
        elif args.processor == 'zero_luma':
            processor = zero_luma_residuals
        elif args.processor == 'zero_chroma':
            processor = zero_chroma_residuals
        elif args.processor == 'scale_residuals':
            processor = scale_residuals(args.scale)
        elif args.processor == 'quantize_residuals':
            processor = quantize_residuals(args.levels)
        elif args.processor == 'noise_residuals':
            processor = add_residual_noise(args.noise_stddev)
        elif args.processor == 'zero_all':
            processor = zero_all
        elif args.processor == 'randomize_intra':
            processor = randomize_intra_modes
        elif args.processor == 'shift_qp':
            processor = shift_qp(args.qp_delta)
        elif args.processor == 'swap_chroma':
            processor = swap_chroma_channels

        pipeline.process_video(
            input_h264=args.input,
            processor=processor,
            output_mp4=args.output,
            keep_intermediates=args.keep,
            intermediate_dir=args.intermediate_dir,
            crf=args.crf
        )


if __name__ == '__main__':
    main()
