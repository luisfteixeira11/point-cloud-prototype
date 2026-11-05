# Gerador de nuvens de pontos

Aplicação de linha de comando que estima a profundidade de uma imagem com
[Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
e exporta uma nuvem de pontos colorida no formato PLY usando Open3D.

## Requisitos

- Python 3.10 ou superior
- Dependências listadas em `requirements.txt`
- Conexão com a internet na primeira execução para baixar o modelo do Hugging Face

## Instalação

No Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No macOS ou Linux, ative o ambiente com `source .venv/bin/activate`.

## Uso

```powershell
python pc.py images\image.jpg --saida pointcloudfile\nuvem.ply
```

O comando mantém os padrões existentes: amostragem de 4 pixels e abertura da
visualização ao final. Para processar sem interface gráfica:

```powershell
python pc.py images\image.jpg --saida pointcloudfile\nuvem.ply --sem-visualizacao
```

### Opções

| Opção | Descrição |
| --- | --- |
| `imagem` | Caminho para a imagem de entrada |
| `--saida` | Destino do arquivo `.ply` (padrão: `nuvem.ply`) |
| `--passo` | Espaçamento de amostragem; valores maiores produzem menos pontos (padrão: `4`) |
| `--sem-visualizacao` | Não abre a janela do Open3D após salvar |

Use `python pc.py --help` para consultar a ajuda no terminal. Uma primeira
execução pode levar mais tempo devido ao download e carregamento do modelo.

## Testes automatizados

Os testes verificam partes importantes do programa, como a normalização da
profundidade, a criação da nuvem de pontos e a leitura e gravação do arquivo
PLY. Eles ajudam a detectar regressões quando o código for alterado e não são
necessários para executar o gerador.

Com o ambiente virtual ativado e a partir da pasta raiz do projeto, execute:

```powershell
python -m unittest discover -s tests -v
```

O comando procura e executa os testes da pasta `tests`. Ao final, o terminal
mostra quais passaram; se algum falhar, exibe os detalhes para ajudar a
investigar o problema. Os testes usam as dependências do projeto, mas não
baixam nem executam o modelo de estimativa de profundidade.