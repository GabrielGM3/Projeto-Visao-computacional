# Projeto Final de Visão Computacional

Projeto acadêmico de Engenharia da Computação sobre transferência de aprendizagem para classificação de imagens de satélite.

## Estado atual

As etapas de carregamento da base, divisão fixa, preparação das imagens, treinamento inicial (Etapa 3), ajuste fino (Etapa 4) e avaliação (Etapa 5) estão concluídas e documentadas em `docs/`. O relatório no formato IEEE (Etapa 6) está em `relatorio/`.

## Proposta

Utilizar a base **EuroSAT RGB** e a **ResNet18 pré-treinada no ImageNet**, disponível no Torchvision, para classificar uso e cobertura do solo em 10 classes.

O objetivo é comparar dois experimentos simples:

1. Treinar apenas a camada final, mantendo o restante da rede congelado.
2. Ajustar também o último bloco da rede, além da camada final.

A comparação prevista inclui acurácia, F1 macro, matriz de confusão e exemplos de erros. As configurações de treinamento serão definidas durante a implementação.

## Escopo

O foco é um projeto de graduação com código simples e explicável. Vamos adaptar um modelo existente, sem desenvolver arquitetura própria, aplicativo ou site.

## Estrutura

```text
.
├── src/              # Modelo e treinamento inicial
├── notebooks/        # Módulos de dados e preparação das imagens; notebooks de uso
├── docs/             # Divisão fixa, documentação e figuras
├── relatorio/        # Relatório IEEE (LaTeX e PDF)
├── tests/            # Verificações de dados e pré-processamento sem download
├── requirements.txt  # Dependências iniciais propostas
└── README.md
```

## Executar as etapas de dados

Na raiz do repositório, instale as dependências em um ambiente virtual:

```bash
python -m pip install -r requirements.txt
```

Baixe a base explicitamente (as imagens ficam em `data/`, ignorada pelo Git):

```bash
python -c "from notebooks.dataset_py import baixar_dataset; print(baixar_dataset())"
```

Confira um lote preparado:

```bash
python notebooks/preprocessamento_py.py
```

Para exploração e figuras, abra `notebooks/02_preprocessamento.ipynb`. O notebook também pode ser executado no Colab. A divisão oficial em `docs/data_split.json` deve ser compartilhada entre os experimentos, sem novos sorteios.

Para executar as verificações automatizadas sem baixar a base:

```bash
python -m unittest discover -s tests -v
```

## Treinamento inicial

Com a base ja baixada, execute na raiz do projeto:

```bash
python -m src.treinamento --num-workers 2
```

O comando treina apenas a camada final por 3 epocas. O melhor modelo e escolhido pela acuracia de validacao. Os pesos e o historico ficam em `outputs/treinamento_inicial/`, fora do Git.

Detalhes e resultados: [Etapa 3](docs/etapa3_treinamento.md).

## Ajuste fino (Etapa 4)

O segundo experimento reutiliza a divisão fixa (`docs/data_split.json`) e o pré-processamento da Etapa 2, descongelando o último bloco convolucional (`layer4`) junto com a camada final (`fc`) — totalizando 8.398.858 parâmetros treináveis (75,11% da rede). Os blocos iniciais (`conv1` a `layer3`) e suas estatísticas de *BatchNorm* permanecem congelados em modo `.eval()`.

O treinamento parte do melhor checkpoint da Etapa 3 (*warm start*) por 5 épocas, com lote de 64 imagens, otimizador Adam, taxa de aprendizado reduzida (`0.0001`), `CrossEntropyLoss` e semente 42.

Com a base baixada e o checkpoint da Etapa 3 em `outputs/treinamento_inicial/melhor_modelo.pth`, execute na raiz do repositório:

```bash
python -m src.ajuste_fino --modelo-inicial outputs/treinamento_inicial/melhor_modelo.pth --epocas 5 --taxa-aprendizado 0.0001 --num-workers 0
```

Também é possível executar partindo diretamente dos pesos do ImageNet (sem *warm start*) omitindo o argumento `--modelo-inicial`.

- **Saídas locais:** `outputs/ajuste_fino/melhor_modelo.pth` (pesos da melhor época na validação) e `outputs/ajuste_fino/resultados.json`.
- **Arquivos versionados:** `docs/etapa4_ajuste_fino.md`, `docs/resultados_ajuste_fino.json` e `docs/figuras/ajuste_fino.png`.
- **Resultado na validação:** melhor desempenho na **época 4**, com **97,63%** de acurácia e perda de **0,0782** — ganho de +4,32 pontos percentuais sobre o checkpoint inicial (redução relativa de 64,6% nos erros de validação). Na época 5 surgiu início de sobreajuste (97,41% e perda 0,0925), descartado automaticamente pelo critério de seleção na validação.

Como os arquivos `.pth` ficam fora do Git, os checkpoints das Etapas 3 e 4 devem ser compartilhados separadamente com o responsável pela Etapa 5.

## Relatório (Etapa 6)

O relatório está em `relatorio/relatorio.pdf`, no template IEEE de conferência em duas colunas e papel A4. O texto reúne a descrição da base, o pré-processamento, os dois experimentos e a avaliação no conjunto de teste, usando os números de `docs/resultados_*.json` e as figuras de `docs/figuras/`. As referências estão em `relatorio/referencias.bib`.

Para gerar o PDF novamente a partir do código-fonte, entre na pasta `relatorio/` e compile com o [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd relatorio
tectonic relatorio.tex
```

Também é possível subir `relatorio.tex`, `referencias.bib` e a pasta `docs/figuras/` no Overleaf (compilador pdfLaTeX). Nesse caso, ajuste o `\graphicspath` no início do `.tex` para o local das figuras.

## Divisão do trabalho

Grupo com 7 integrantes, conforme autorização do professor.

| Função | Responsabilidade | Entrega prevista |
| --- | --- | --- |
| 1 — Base de dados | Obter e explorar a base; conferir classes e quantidades; preparar uma divisão fixa de treino, validação e teste. | Carregamento, divisão salva e descrição da base. |
| 2 — Preparação das imagens | Definir redimensionamento, normalização e pequenas variações para treino; conferir os lotes. | Preparação das imagens e exemplos visualizados. |
| 3 — Modelo e treinamento inicial — Guilherme | Adaptar a saída da ResNet18 e treinar apenas a camada final. | Primeiro experimento, modelo salvo e histórico. |
| 4 — Ajuste fino | Reutilizar a estrutura de treinamento para ajustar também o último bloco da rede. | Segundo experimento e registro das configurações. |
| 5 — Avaliação | Comparar os modelos na mesma base de teste e analisar os erros. | Métricas, gráficos e análise dos resultados. |
| 6 — Relatório e referências | Organizar as contribuições no template IEEE e revisar o texto. | Relatório em PDF, pronto para impressão. |
| 7 — Apresentação e reprodução | Organizar slides e instruções de execução; conferir a reprodução do projeto. | Apresentação e README atualizado. |

Cada integrante deve escrever sobre sua parte e contribuir com a apresentação. Todos devem compreender o fluxo completo para a arguição.

## Próximos passos

- Preencher nomes e instituição no início de `relatorio/relatorio.tex` e gerar o PDF final.
- Imprimir o relatório (mesma versão do PDF entregue no Canvas).
- Preparar a apresentação a partir das figuras e tabelas do relatório.

## Dependências

O arquivo `requirements.txt` lista as bibliotecas previstas. A revisão das etapas de dados foi executada em CPU, no Windows, com Python 3.14, PyTorch 2.14.1 e Torchvision 0.29.2. A configuração de GPU será definida nas etapas de treinamento.

## Entregas previstas no enunciado

- Código e relatório digital no Canvas até **25/11/2026**, sem enviar a base de dados.
- Relatório em formato de artigo IEEE de duas colunas, com versões digital e impressa iguais; usar A4 para a versão em Word.
- Agendamento presencial da apresentação até **25/11/2026**, após a entrega do relatório impresso.
- Apresentações até **09/12/2026**.

## Referências iniciais

- [EuroSAT — repositório oficial](https://github.com/phelber/EuroSAT)
- [ResNet18 — documentação do Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [Templates de conferência IEEE](https://www.ieee.org/conferences/publishing/templates)
