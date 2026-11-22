# -*- coding: utf-8 -*-
"""preprocessamento_py.py

Etapa 2 — Preparação das imagens (EuroSAT RGB + ResNet18 pré-treinada no ImageNet).

Este módulo NÃO refaz a divisão dos dados. Ele reutiliza `obter_datasets()` e
`EuroSATSubset` do `dataset_py.py` (Etapa 1) e apenas acrescenta:
  - as transformações de cada partição (treino x validação/teste);
  - a criação dos DataLoaders;
  - funções para conferir e visualizar os lotes.

O download é explícito, via `baixar_dataset()` do módulo `dataset_py.py`.
Execute a partir da raiz do repositório para usar os caminhos padrão.
"""

import os
import random

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# Funciona tanto importando como "notebooks.preprocessamento_py" (padrão do
# InitEG.ipynb, executado a partir da raiz) quanto de dentro da pasta notebooks/.
try:
    from notebooks.dataset_py import obter_datasets, EuroSATSubset  # noqa: F401
except ModuleNotFoundError:
    from dataset_py import obter_datasets, EuroSATSubset  # noqa: F401


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Média e desvio padrão (por canal R, G, B) das imagens do ImageNet.
# A ResNet18 foi pré-treinada com entradas normalizadas exatamente com esses
# valores, então usamos os mesmos para que as imagens "conversem" com os pesos.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Tamanho de entrada da rede. As imagens da EuroSAT têm 64x64 e são ampliadas
# para 224x224, a resolução usada no pré-treino da ResNet18.
# É um parâmetro: as funções abaixo aceitam outro valor (ex.: 64 ou 128) para
# testes mais rápidos.
IMG_SIZE = 224

# Caminhos padrão (relativos à raiz do repositório)
CAMINHO_DADOS = "./data/eurosat/2750"
CAMINHO_SPLIT = "docs/data_split.json"


# ---------------------------------------------------------------------------
# Transformações
# ---------------------------------------------------------------------------

class RotacaoMultiplos90:
    """Gira a imagem em 0°, 90°, 180° ou 270°, escolhido ao acaso.

    Imagens de satélite são vistas "de cima": não existe um lado certo para
    cima, então girar em múltiplos de 90° gera uma imagem igualmente válida.
    Como a imagem é quadrada, o giro não corta nada nem cria bordas pretas
    (diferente de uma rotação por um ângulo qualquer, como 30°).
    """

    def __init__(self, angulos=(0, 90, 180, 270)):
        self.angulos = angulos

    def __call__(self, img):
        angulo = random.choice(self.angulos)
        return TF.rotate(img, angulo)

    def __repr__(self):
        return f"RotacaoMultiplos90(angulos={self.angulos})"


def get_transforms_treino(img_size=IMG_SIZE):
    """Transformações do TREINO: redimensiona + pequenas variações aleatórias."""
    return transforms.Compose([
        # 64x64 -> img_size x img_size
        transforms.Resize((img_size, img_size)),
        # Espelhamentos: continuam sendo imagens de satélite válidas
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # Giro de 0/90/180/270 graus
        RotacaoMultiplos90(),
        # Variação LEVE de brilho, contraste e saturação (simula diferenças de
        # iluminação/atmosfera). Não alteramos o matiz (hue), pois a cor é uma
        # pista importante (ex.: água azul, vegetação verde).
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        # PIL (0..255) -> tensor float (0..1) no formato [C, H, W]
        transforms.ToTensor(),
        # Normalização com as estatísticas do ImageNet
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_transforms_avaliacao(img_size=IMG_SIZE):
    """Transformações de VALIDAÇÃO e TESTE: sem nada aleatório (determinístico)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def obter_dataloaders(data_dir=CAMINHO_DADOS, img_size=IMG_SIZE, batch_size=64,
                      num_workers=2, split_json_path=CAMINHO_SPLIT, seed=42):
    """Cria os DataLoaders de treino, validação e teste.

    Retorna: (treino_dl, val_dl, teste_dl, classes)
      - classes: lista de nomes na ordem dos índices do "class_to_idx" do
        data_split.json (classes[i] é o nome da classe de rótulo i).

    Só o treino é embaralhado (shuffle=True). Validação e teste mantêm sempre a
    mesma ordem, para que os resultados sejam comparáveis entre experimentos.
    O parâmetro `seed` fixa a ordem do embaralhamento do treino.
    """
    # Proteção: se o JSON não for encontrado, o dataset_py.py criaria uma NOVA
    # divisão e sobrescreveria o arquivo. Aqui preferimos parar com um erro claro.
    if not os.path.exists(split_json_path):
        raise FileNotFoundError(
            f"Não encontrei '{split_json_path}'. Execute a partir da raiz do "
            "repositório (a divisão oficial não deve ser recriada)."
        )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Pasta de imagens não encontrada: '{data_dir}'.")

    # Datasets da Etapa 1 (mesma divisão fixa para todos do grupo)
    treino_ds, val_ds, teste_ds = obter_datasets(data_dir, split_json_path)

    # Cada partição recebe a sua própria transformação
    treino_ds.transform = get_transforms_treino(img_size)
    val_ds.transform = get_transforms_avaliacao(img_size)
    teste_ds.transform = get_transforms_avaliacao(img_size)

    # Nomes das classes vindos do class_to_idx (EuroSATSubset já ordena por índice)
    classes = treino_ds.classes

    # pin_memory acelera a cópia para a GPU quando ela existe
    usar_pin = torch.cuda.is_available()

    # Gerador com semente: mesma ordem de embaralhamento a cada execução
    gerador = torch.Generator()
    gerador.manual_seed(seed)

    treino_dl = DataLoader(treino_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=usar_pin,
                           generator=gerador)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=usar_pin)
    teste_dl = DataLoader(teste_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=usar_pin)

    return treino_dl, val_dl, teste_dl, classes


# ---------------------------------------------------------------------------
# Conferência e visualização
# ---------------------------------------------------------------------------

def desnormalizar(tensor):
    """Desfaz a normalização do ImageNet para conseguirmos visualizar a imagem.

    Aceita uma imagem [3, H, W] ou um lote [B, 3, H, W].
    Retorna valores entre 0 e 1 (prontos para o plt.imshow após permute).
    """
    media = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    desvio = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    imagem = tensor.cpu() * desvio + media
    return imagem.clamp(0, 1)


def conferir_lote(dataloader, classes):
    """Pega UM lote e imprime informações para conferência.

    - shape e dtype das imagens e dos rótulos;
    - faixa dos rótulos (devem estar entre 0 e len(classes)-1);
    - média e desvio padrão por canal (após a normalização);
    - quantas imagens de cada classe vieram no lote.
    """
    imagens, rotulos = next(iter(dataloader))

    print(f"Imagens: shape={tuple(imagens.shape)}  dtype={imagens.dtype}")
    print(f"Rótulos: shape={tuple(rotulos.shape)}  dtype={rotulos.dtype}")
    print(f"Faixa dos rótulos: min={rotulos.min().item()}  max={rotulos.max().item()}"
          f"  (válido: 0 a {len(classes) - 1})")
    print(f"Faixa dos pixels (normalizados): min={imagens.min().item():.3f}"
          f"  max={imagens.max().item():.3f}")

    # Média e desvio por canal: calculados sobre lote, altura e largura
    medias = imagens.mean(dim=(0, 2, 3))
    desvios = imagens.std(dim=(0, 2, 3))
    for i, canal in enumerate(["R", "G", "B"]):
        print(f"  Canal {canal}: média={medias[i].item():+.3f}  desvio={desvios[i].item():.3f}")

    # Contagem de classes no lote
    contagem = torch.bincount(rotulos, minlength=len(classes))
    print("Contagem de classes no lote:")
    for i, nome in enumerate(classes):
        if contagem[i] > 0:
            print(f"  [{i}] {nome:<22} {contagem[i].item()}")

    return imagens, rotulos


def _salvar_figura(fig, salvar_em):
    """Salva a figura (criando a pasta, se preciso)."""
    if salvar_em:
        pasta = os.path.dirname(salvar_em)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        fig.savefig(salvar_em, dpi=120, bbox_inches="tight")
        print(f"Figura salva em: {salvar_em}")


def plotar_grade_lote(dataloader, classes, salvar_em=None, n_imagens=16):
    """Mostra uma grade com as primeiras `n_imagens` de um lote e suas classes."""
    imagens, rotulos = next(iter(dataloader))
    n = min(n_imagens, len(imagens))
    n_colunas = 4
    n_linhas = (n + n_colunas - 1) // n_colunas

    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(3 * n_colunas, 3 * n_linhas))
    eixos = eixos.flatten()
    for i, eixo in enumerate(eixos):
        eixo.axis("off")
        if i < n:
            # [3, H, W] -> [H, W, 3] para o matplotlib
            eixo.imshow(desnormalizar(imagens[i]).permute(1, 2, 0).numpy())
            eixo.set_title(f"{rotulos[i].item()}: {classes[rotulos[i]]}", fontsize=10)

    fig.tight_layout()
    _salvar_figura(fig, salvar_em)
    plt.show()


def plotar_aumentos(dataset_treino, idx, n=5, salvar_em=None):
    """Mostra a imagem original (64x64) e `n` versões geradas pelas
    transformações aleatórias do treino, para a mesma amostra `idx`."""
    if dataset_treino.transform is None:
        raise ValueError("O dataset não tem transformação. Use treino_dl.dataset.")

    caminho, rotulo = dataset_treino.samples[idx]
    with Image.open(caminho) as img:
        original = img.convert("RGB")

    fig, eixos = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3.4))

    eixos[0].imshow(original)
    eixos[0].set_title(f"Original {original.size[0]}x{original.size[1]}")
    eixos[0].axis("off")

    for i in range(1, n + 1):
        # Cada chamada sorteia novas variações
        aumentada = dataset_treino.transform(original)
        eixos[i].imshow(desnormalizar(aumentada).permute(1, 2, 0).numpy())
        eixos[i].set_title(f"Aumento {i} ({aumentada.shape[1]}x{aumentada.shape[2]})")
        eixos[i].axis("off")

    nome_classe = dataset_treino.classes[rotulo]
    fig.suptitle(f"Classe: {nome_classe} — {os.path.basename(caminho)}", fontsize=12)
    fig.tight_layout()
    _salvar_figura(fig, salvar_em)
    plt.show()


if __name__ == "__main__":
    # Teste rápido: python notebooks/preprocessamento_py.py (a partir da raiz)
    treino_dl, val_dl, teste_dl, classes = obter_dataloaders(num_workers=0)
    print(f"Classes: {classes}")
    print(f"Lotes -> treino: {len(treino_dl)} | validação: {len(val_dl)} | teste: {len(teste_dl)}")
    conferir_lote(treino_dl, classes)
