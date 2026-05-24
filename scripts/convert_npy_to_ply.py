#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert NumPy point-cloud files (.npy) to PLY files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="examples/kitti",
        help="Input .npy file or directory containing .npy files. Defaults to examples/kitti.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for converted .ply files. Defaults to the input file's parent directory.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write ASCII PLY instead of binary little-endian PLY.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .ply files.",
    )
    return parser.parse_args()


def discover_npy_files(input_path):
    input_path = Path(input_path).expanduser()
    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            raise ValueError(f"Expected a .npy file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.npy"))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def normalize_points(array, path):
    points = np.asarray(array)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path} must have shape (N, 3+) but has {points.shape}")

    points = points[:, :3]
    finite_mask = np.isfinite(points).all(axis=1)
    if not finite_mask.all():
        skipped = int((~finite_mask).sum())
        print(f"Skipping {skipped} non-finite points from {path}")
        points = points[finite_mask]

    return np.ascontiguousarray(points, dtype="<f4")


def write_ply(path, points, ascii_format=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0" if ascii_format else "format binary_little_endian 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
            "",
        ]
    )

    if ascii_format:
        with path.open("w", encoding="ascii") as file:
            file.write(header)
            np.savetxt(file, points, fmt="%.8f")
    else:
        with path.open("wb") as file:
            file.write(header.encode("ascii"))
            points.tofile(file)


def convert_file(input_path, output_dir=None, ascii_format=False, overwrite=False):
    output_root = output_dir if output_dir is not None else input_path.parent
    output_path = output_root / f"{input_path.stem}.ply"
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return output_path

    points = normalize_points(np.load(input_path, allow_pickle=False), input_path)
    write_ply(output_path, points, ascii_format=ascii_format)
    print(f"Wrote {output_path} ({len(points)} points)")
    return output_path


def main():
    args = parse_args()
    npy_files = discover_npy_files(args.input)
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under: {args.input}")

    output_dir = args.output_dir.expanduser() if args.output_dir else None
    for npy_file in npy_files:
        convert_file(
            npy_file,
            output_dir=output_dir,
            ascii_format=args.ascii,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
