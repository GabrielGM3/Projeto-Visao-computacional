# Etapa 2 — Preparação das imagens

Arquivos desta etapa:

- `notebooks/preprocessamento_py.py`: módulo com as transformações, os DataLoaders e as funções de conferência e visualização.
- `notebooks/02_preprocessamento.ipynb`: notebook que roda tudo, confere os lotes e gera as figuras.
- `docs/figuras/`: `lote_treino.png`, `aumentos.png` e `lote_validacao.png`.

A divisão dos dados **não** é refeita nesta etapa. Usamos `obter_datasets()` e `EuroSATSubset` do `notebooks/dataset_py.py` (Etapa 1), que leem a divisão fixa salva em `docs/data_split.json`. Os nomes e índices das classes vêm do `class_to_idx` desse arquivo:

| Índice | Classe | Índice | Classe |
| :---: | :--- | :---: | :--- |
| 0 | AnnualCrop | 5 | Pasture |
| 1 | Forest | 6 | PermanentCrop |
| 2 | HerbaceousVegetation | 7 | Residential |
| 3 | Highway | 8 | River |
| 4 | Industrial | 9 | SeaLake |

## 1. Decisões e justificativas

### Por que redimensionar para 224×224?

As imagens da EuroSAT têm 64×64 pixels, mas a ResNet18 foi pré-treinada no ImageNet com entradas de 224×224. Os filtros das primeiras camadas aprenderam a reconhecer bordas e texturas nessa escala. Além disso, a ResNet18 reduz a resolução 32 vezes até a última camada convolucional: com 64×64 sobraria um mapa de apenas 2×2, contra 7×7 com 224×224. Ampliar a imagem não acrescenta informação nova, mas deixa os objetos (casas, estradas, rios) num tamanho próximo daquele que a rede aprendeu a reconhecer, o que costuma melhorar a transferência de aprendizagem.

O custo é computacional: cada imagem fica cerca de 12 vezes maior. Por isso o tamanho é um parâmetro (`img_size`). Para testes rápidos, é possível usar 64 ou 128, mas os experimentos oficiais das Etapas 3 e 4 devem usar o mesmo valor, de preferência 224.

### Por que usar a normalização do ImageNet?

A rede pré-treinada espera entradas normalizadas com a média `[0.485, 0.456, 0.406]` e o desvio `[0.229, 0.224, 0.225]` (canais R, G, B) do ImageNet, porque foi assim que ela foi treinada. Se usássemos outra normalização, as entradas teriam outra distribuição e os pesos aprendidos deixariam de "encaixar". Isso importa principalmente no Experimento 1 da Etapa 3, em que quase toda a rede fica congelada.

Consequência: depois de normalizado, um lote da EuroSAT **não** fica com média 0 e desvio 1 em cada canal, porque as imagens de satélite têm uma distribuição de cores diferente da do ImageNet. No lote de treino conferido no notebook, as médias foram R ≈ −0,55, G ≈ −0,31 e B ≈ 0,01, e os desvios foram ≈ 0,96, 0,67 e 0,56. Isso é esperado e não é um erro. O que importa é que a rede receba os dados na mesma escala usada no pré-treino. Calcular a média e o desvio da própria EuroSAT seria uma alternativa, mas afastaria as entradas daquilo que os pesos congelados esperam.

### Por que essas variações (e não recortes)?

As variações do treino precisam produzir imagens que **continuem sendo exemplos válidos da mesma classe**:

- **Espelhamentos horizontal e vertical e giros de 0°, 90°, 180° e 270°:** a imagem de satélite é vista de cima e não tem um lado certo para cima. Um rio girado continua sendo um rio. Como as imagens são quadradas, giros em múltiplos de 90° não cortam nada e não criam bordas pretas, ao contrário de giros por ângulos quaisquer. Juntas, essas operações geram até 8 orientações diferentes de cada imagem.
- **ColorJitter leve (brilho, contraste e saturação ±10%):** simula pequenas diferenças de iluminação, de atmosfera e de época de captura. Mantemos a variação pequena e **não alteramos o matiz (hue)**, porque a cor é uma pista importante para várias classes, como `SeaLake` (azul), `Forest` (verde-escuro) e `AnnualCrop` (tons de terra).
- **Sem recortes aleatórios (RandomCrop / RandomResizedCrop):** as imagens têm só 64×64 pixels, ou seja, cada uma cobre uma área pequena (cerca de 640 m × 640 m). Um recorte descartaria uma parte grande do conteúdo e poderia remover justamente o elemento que define a classe, como a estrada em `Highway` ou o rio em `River`. O `RandomResizedCrop` ainda mudaria a escala aparente, que é uma informação útil nas imagens de satélite (todas foram captadas com a mesma resolução, 10 m por pixel).

Validação e teste **não recebem nenhuma variação aleatória**. Assim, a mesma imagem gera sempre o mesmo tensor e os resultados dos dois experimentos podem ser comparados de forma justa. O notebook comprova isso carregando a mesma imagem duas vezes e conferindo com `torch.equal` que os tensores são idênticos.

### Outras decisões

- **Só o treino é embaralhado.** O `data_split.json` está ordenado por classe. Sem o shuffle, cada lote de treino teria uma única classe, o que prejudica o treinamento. Em validação e teste a ordem fixa não atrapalha, porque as métricas são calculadas sobre a partição inteira. A consequência é que o primeiro lote do `val_dl` contém só `AnnualCrop`.
- **Figura de validação com amostra fixa.** Para que `lote_validacao.png` mostre várias classes, o notebook sorteia uma amostra fixa de 64 imagens da validação (`random.Random(42)`), monta um `Subset` e cria um DataLoader usado só para a figura. Essas imagens recebem as mesmas transformações de avaliação. A amostra serve apenas para visualização: o `val_dl` oficial continua sem embaralhamento, e as métricas usam a partição inteira.
- **Figuras reproduzíveis.** As células que geram `lote_treino.png` e `aumentos.png` fixam as sementes (`random.seed(0)` e `torch.manual_seed(0)`), para que as variações aleatórias saiam iguais a cada execução do notebook. Isso vale só para as figuras e não muda o treinamento.
- **Semente no embaralhamento (`seed=42`).** A ordem dos lotes de treino se repete em cada execução, o que facilita comparar os experimentos 1 e 2. As variações aleatórias de cada imagem continuam diferentes a cada época.
- **Proteção do split.** Se o `docs/data_split.json` não for encontrado (por exemplo, quando o código é executado fora da raiz do repositório), `obter_dataloaders()` para com um erro em vez de deixar o `dataset_py.py` criar uma nova divisão.

## 2. Transformações por partição

| Ordem | Transformação | Treino | Validação | Teste | Observação |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `Resize((224, 224))` | ✔ | ✔ | ✔ | 64×64 → 224×224 |
| 2 | `RandomHorizontalFlip(p=0.5)` | ✔ | — | — | espelhamento lateral |
| 3 | `RandomVerticalFlip(p=0.5)` | ✔ | — | — | espelhamento vertical |
| 4 | `RotacaoMultiplos90()` | ✔ | — | — | sorteia 0°, 90°, 180° ou 270° |
| 5 | `ColorJitter(0.1, 0.1, 0.1)` | ✔ | — | — | brilho, contraste e saturação ±10%; sem hue |
| 6 | `ToTensor()` | ✔ | ✔ | ✔ | PIL 0–255 → tensor `float32` 0–1, `[C, H, W]` |
| 7 | `Normalize(IMAGENET_MEAN, IMAGENET_STD)` | ✔ | ✔ | ✔ | estatísticas do ImageNet |
| — | Embaralhamento (`shuffle`) | ✔ | — | — | ordem fixa por `seed=42` |

Saída de cada lote: imagens `[64, 3, 224, 224]` em `float32` e rótulos `[64]` em `int64`, com valores de 0 a 9. Quantidades: treino com 18.900 imagens (296 lotes), validação com 4.050 (64 lotes) e teste com 4.050 (64 lotes).

## 3. Como usar nas próximas etapas

Execute sempre a partir da **raiz do repositório**, porque os caminhos `./data` e `docs/` são relativos a ela.

> **Efeito colateral:** importar `preprocessamento_py` também importa o `dataset_py.py`, que no topo do arquivo executa `datasets.EuroSAT(root='./data', download=True)`. Na primeira execução isso baixa a base, que tem cerca de 90 MB. Nas seguintes, só confere se ela já existe. Se o import for feito a partir de outra pasta, a base será baixada em outro lugar.

### Etapas 3 e 4 — Treinamento e ajuste fino

```python
import sys, os
sys.path.append(os.path.abspath("notebooks"))

import torch
from notebooks.preprocessamento_py import obter_dataloaders

treino_dl, val_dl, teste_dl, classes = obter_dataloaders(
    data_dir="./data/eurosat/2750",
    img_size=224,
    batch_size=64,
    num_workers=2,      # no Windows local, usar 0
)

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(classes)   # 10 -> tamanho da nova camada final da ResNet18

for epoca in range(num_epocas):
    modelo.train()
    for imagens, rotulos in treino_dl:
        imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)
        ...  # forward, loss, backward, step

    modelo.eval()
    with torch.no_grad():
        for imagens, rotulos in val_dl:
            ...  # acurácia de validação para escolher o melhor modelo
```

Os dois experimentos devem usar os mesmos `img_size` e `batch_size` para que a comparação seja justa.

### Etapa 5 — Avaliação

```python
treino_dl, val_dl, teste_dl, classes = obter_dataloaders(data_dir="./data/eurosat/2750")

modelo.eval()
todos_rotulos, todas_predicoes = [], []
with torch.no_grad():
    for imagens, rotulos in teste_dl:          # só o teste
        saidas = modelo(imagens.to(dispositivo))
        todas_predicoes.extend(saidas.argmax(1).cpu().tolist())
        todos_rotulos.extend(rotulos.tolist())

# Como o teste não é embaralhado, a i-ésima predição corresponde a
# teste_dl.dataset.samples[i] (caminho da imagem), o que ajuda na análise de erros.
# Para nomear as classes no relatório e na matriz de confusão: classes[indice]
```

Para mostrar uma imagem do lote (por exemplo, um erro do modelo), use `desnormalizar(tensor)` antes do `plt.imshow`:

```python
from notebooks.preprocessamento_py import desnormalizar
plt.imshow(desnormalizar(imagens[0]).permute(1, 2, 0))
```
