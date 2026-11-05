"""Generate a colored point cloud from an image using Depth Anything V2."""

import argparse
from pathlib import Path
from typing import Sequence
import numpy as np
import open3d as o3d
from PIL import Image

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Estima a profundidade de uma imagem e salva uma nuvem .ply."
    )
    parser.add_argument("imagem", type=Path, help="Caminho para a imagem de entrada.")
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("nuvem.ply"),
        help="Arquivo .ply de saída (padrão: nuvem.ply).",
    )
    parser.add_argument(
        "--passo",
        type=int,
        default=4,
        help="Espaçamento da amostragem em pixels; valores maiores geram menos pontos (padrão: 4).",
    )
    parser.add_argument(
        "--sem-visualizacao",
        action="store_true",
        help="Salva o arquivo sem abrir a janela de visualização.",
    )
    args = parser.parse_args(argv)

    if not args.imagem.is_file():
        parser.error(f"imagem não encontrada: {args.imagem}")
    if args.passo < 1:
        parser.error("--passo deve ser maior que zero.")
    if args.saida.suffix.lower() != ".ply":
        parser.error("o arquivo de saída deve ter a extensão .ply.")

    return args


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize a depth map to the 0–1 range."""
    if depth.ndim != 2 or not depth.size:
        raise ValueError("O mapa de profundidade deve ser uma matriz 2D não vazia.")
    if not np.isfinite(depth).all():
        raise ValueError("O mapa de profundidade contém valores não finitos.")

    depth_min = float(depth.min())
    depth_range = float(depth.max()) - depth_min
    if depth_range == 0:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - depth_min) / depth_range).astype(np.float32)


def create_point_cloud(
    image: Image.Image, depth: np.ndarray, step: int
) -> o3d.geometry.PointCloud:
    """Create an RGB point cloud from an image, a depth map, and a pixel step."""
    if step < 1:
        raise ValueError("O passo de amostragem deve ser maior que zero.")

    depth = normalize_depth(depth)
    height, width = depth.shape
    y, x = np.mgrid[0:height:step, 0:width:step]
    sampled_depth = depth[::step, ::step]
    scale = max(height, width)
    points = np.column_stack(
        (
            ((x - width / 2) / scale).ravel(),
            (-(y - height / 2) / scale).ravel(),
            sampled_depth.ravel(),
        )
    )

    resized_image = image.convert("RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    colors = np.asarray(resized_image, dtype=np.float64)[::step, ::step]
    colors = colors.reshape(-1, 3) / 255.0

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def estimate_depth(image: Image.Image) -> np.ndarray:
    """Estimate image depth with the configured Hugging Face model."""
    from transformers import pipeline

    estimator = pipeline(
        "depth-estimation",
        model=MODEL_ID,
        device=-1,
    )
    result = estimator(image)
    return np.asarray(result["depth"], dtype=np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    with Image.open(args.imagem) as source:
        image = source.convert("RGB")

    depth = estimate_depth(image)
    point_cloud = create_point_cloud(image, depth, args.passo)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.saida), point_cloud):
        raise RuntimeError(f"Não foi possível salvar a nuvem em {args.saida}.")

    print(f"Nuvem salva em: {args.saida.resolve()}")
    print(f"Quantidade de pontos: {len(point_cloud.points):,}")
    if not args.sem_visualizacao:
        o3d.visualization.draw_geometries([point_cloud])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())