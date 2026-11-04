import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import pipeline


def main():
    # parser e um modulo pra criacao de interfaces por linha de comando
    parser = argparse.ArgumentParser(
        description="Estima profundidade de uma foto e cria uma nuvem de pontos .ply."
    )
    parser.add_argument("imagem", type=Path, help="Caminho para a foto de entrada")
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("nuvem.ply"),
        help="Arquivo .ply de saida (padrao: nuvem.ply)",
    )
    parser.add_argument(
        "--passo",
        type=int,
        default=4,
        help="Amostragem dos pixels; aumente para gerar menos pontos (padrao: 4)",
    )
    args = parser.parse_args()

    if not args.imagem.is_file():
        parser.error(f"arquivo de imagem nao encontrado: {args.imagem}")
    if args.passo < 1:
        parser.error("--passo deve ser maior que zero")

    # conversao pro rgb pra utilizacao de imagem na forma matricial
    imagem = Image.open(args.imagem).convert("RGB")

    # cria a pipeline do modelo 
    estimador = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=-1,
    )

    # matriz numpy com a estimativa do modelo da nuvem de pontos
    profundidade = np.asarray(estimador(imagem)["depth"], dtype=np.float32)
    h, w = profundidade.shape
    profundidade = (profundidade - profundidade.min()) / (
        profundidade.max() - profundidade.min() + 1e-8
    )


    y, x = np.mgrid[0:h:args.passo, 0:w:args.passo]
    z = profundidade[::args.passo, ::args.passo]
    escala = max(h, w)
    pontos = np.column_stack(
        (
            ((x - w / 2) / escala).ravel(),
            (-(y - h / 2) / escala).ravel(),
            z.ravel(),
        )
    )

    # geração da nuvem de pontos com o open3d
    imagem_cores = imagem.resize((w, h), Image.Resampling.BILINEAR)
    cores = (
        np.asarray(imagem_cores)[::args.passo, ::args.passo]
        .reshape(-1, 3)
        .astype(np.float64)
        / 255.0
    )
    nuvem = o3d.geometry.PointCloud()
    nuvem.points = o3d.utility.Vector3dVector(pontos)
    nuvem.colors = o3d.utility.Vector3dVector(cores)

    # criação do arquivo .ply
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.saida), nuvem):
        raise RuntimeError(f"nao foi possivel salvar a nuvem em {args.saida}")

    print(f"Nuvem salva em: {args.saida.resolve()}")
    print(f"Quantidade de pontos: {len(nuvem.points)}")
    o3d.visualization.draw_geometries([nuvem])


if __name__ == "__main__":
    main()