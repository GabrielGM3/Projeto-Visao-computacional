# Entrega 1 — Base de Dados e Divisão Fixa (EuroSAT RGB)

## 1. Resumo das Atividades Realizadas
Nesta primeira etapa do projeto, foi implementado o módulo `notebooks/dataset_py.py`, responsável pela leitura local, validação de integridade, análise estatística, padronização de canais e particionamento reprodutível da base de dados **EuroSAT (versão RGB)**.

Principais entregas implementadas:
- **Sanity Check (`verificar_integridade_imagens`):** Varredura completa via `PIL.Image.verify()` em todos os arquivos locais antes do carregamento, descartando o risco de imagens corrompidas interromperem o treinamento.
- **Padronização de Canais (`convert("RGB")`):** Conversão explícita de todas as amostras para 3 canais RGB no momento da leitura (`__getitem__`), garantindo que eventuais arquivos fora do padrão (como imagens em escala de cinza com 1 canal ou imagens RGBA com canal alfa de transparência) não causem erros de dimensão nos tensores da ResNet18.
- **Divisão Estratificada Fixa (`gerar_ou_carregar_split`):** Particionamento dos dados mantendo exatamente a mesma proporção de classes em cada conjunto (`70% Treino`, `15% Validação`, `15% Teste`) com semente fixa (`seed = 42`).
- **Persistência Dupla em JSON:**
  1. `docs/data_split.json`: Estrutura otimizada para consumo direto pelos scripts do PyTorch, contendo o dicionário `class_to_idx` e os caminhos relativos normalizados (`/`) de cada partição.
  2. `docs/mapeamento_classes_objetivo.json`: Estrutura hierárquica organizada explicitamente por **Classe → Objetivo (treino, validação, teste)** com contagens detalhadas para auditoria rápida e documentação.
- **Classe `EuroSATSubset` Isolada:** Estrutura de `torch.utils.data.Dataset` desacoplada de transformações visuais, evitando vazamento de *data augmentation* entre os conjuntos de treino e validação/teste.
- **Utilitários de Inspeção Manual (`plotar_imagem_aleatoria` e `plotar_imagem_por_path`):** Funções para visualização rápida de amostras no ambiente de execução (como Google Colab) acompanhadas de seus caminhos exatos e rótulos reais.

---

## 2. Análise de Distribuição e Amostragem Estratificada

A base EuroSAT RGB possui **27.000 imagens** (64×64 pixels) distribuídas em **10 classes**, apresentando um **leve desbalanceamento natural** entre suas categorias:

- **Grupo Majoritário (3.000 imagens por classe):** `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Residential` e `SeaLake`.
- **Grupo Intermediário (2.500 imagens por classe):** `Highway`, `Industrial`, `PermanentCrop` e `River`.
- **Grupo Minoritário (2.000 imagens):** `Pasture` (possui 33,3% menos amostras que as classes majoritárias).

### Justificativa Metodológica
Diante dessa variação entre 2.000 e 3.000 amostras por classe, um sorteio puramente aleatório sobre o montante total de 27.000 imagens poderia gerar distorções estatísticas, sub-representando classes menores (como `Pasture`) no conjunto de teste.

A **amostragem estratificada** aplicada resolve esse problema ao agrupar os arquivos por categoria antes de embaralhar e aplicar o corte `70% / 15% / 15%` individualmente dentro de cada classe. Isso garante representatividade homogênea em todas as partições e assegura a confiabilidade do cálculo de **F1 Macro** e da **Matriz de Confusão** nas etapas de avaliação.

### Tabela Completa de Particionamento

| ID | Classe | Total | Treino (70%) | Validação (15%) | Teste (15%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | `AnnualCrop` | 3.000 | 2.100 | 450 | 450 |
| **1** | `Forest` | 3.000 | 2.100 | 450 | 450 |
| **2** | `HerbaceousVegetation` | 3.000 | 2.100 | 450 | 450 |
| **3** | `Highway` | 2.500 | 1.750 | 375 | 375 |
| **4** | `Industrial` | 2.500 | 1.750 | 375 | 375 |
| **5** | `Pasture` | 2.000 | 1.400 | 300 | 300 |
| **6** | `PermanentCrop` | 2.500 | 1.750 | 375 | 375 |
| **7** | `Residential` | 3.000 | 2.100 | 450 | 450 |
| **8** | `River` | 2.500 | 1.750 | 375 | 375 |
| **9** | `SeaLake` | 3.000 | 2.100 | 450 | 450 |
| — | **Total Geral** | **27.000** | **18.900** | **4.050** | **4.050** |

---

## 3. Bibliotecas Utilizadas e Justificativas

O módulo de gerenciamento de dados (`notebooks/dataset_py.py`) faz uso das seguintes bibliotecas e pacotes do ecossistema Python:

| Biblioteca / Módulo | Função no Projeto |
| :--- | :--- |
| **`os`** | Módulo nativo do Python utilizado para manipulação de caminhos de diretórios e normalização de separadores (`/` vs `\`), garantindo portabilidade entre ambientes Windows, Linux e Google Colab. |
| **`json`** | Responsável por serializar e desserializar os arquivos `data_split.json` e `mapeamento_classes_objetivo.json`, garantindo que todos os integrantes do grupo carreguem exatamente a mesma divisão fixa sem novos sorteios. |
| **`random`** | Módulo nativo empregado para aplicar a semente fixa (`seed = 42`) e realizar o embaralhamento determinístico dentro de cada estrato de classe. |
| **`PIL (Pillow)`** | Biblioteca de processamento de imagens acionada tanto no *Sanity Check* (`img.verify()`) quanto na leitura segura de cada amostra com conversão padronizada para 3 canais (`img.convert("RGB")`). |
| **`matplotlib.pyplot`** | Utilizada para a renderização gráfica nas funções de inspeção visual (`plotar_imagem_aleatoria` e `plotar_imagem_por_path`). |
| **`torch` (PyTorch)** | Framework base de *Deep Learning*, fornecendo a classe abstrata `torch.utils.data.Dataset` herdada por `EuroSATSubset`. |
| **`torchvision`** | Extensão de visão computacional utilizada por meio do `datasets.ImageFolder` para mapear automaticamente a árvore de subpastas locais da EuroSAT nos índices numéricos das 10 classes. |

---

## 4. Guia de Integração para as Próximas Etapas
- **Regra de Ouro:** Os arquivos `docs/data_split.json` e `docs/mapeamento_classes_objetivo.json` devem ser versionados no Git (a pasta com as imagens locais deve permanecer no `.gitignore`). Nenhum integrante deve gerar novas divisões aleatórias.
- **Para a Função 2 (Preparação de Imagens):** Chamar `treino_ds, val_ds, teste_ds = obter_datasets("./data/eurosat/2750")` e atribuir o atributo `.transform` individualmente a cada objeto (`treino_ds.transform = ...`), sem risco de vazar *data augmentation* para validação ou teste.
- **Para a Função 5 (Avaliação e Inferência):** Utilizar exclusivamente o `teste_ds` e as funções `plotar_imagem_aleatoria(teste_ds)` ou `plotar_imagem_por_path(...)` para inspecionar predições individuais sem risco de avaliar imagens vistas durante o treinamento.
