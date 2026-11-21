# Projeto Final de Visão Computacional

Projeto acadêmico de Engenharia da Computação sobre transferência de aprendizagem para classificação de imagens de satélite.

## Estado atual

Organização inicial do repositório. A implementação, o download da base, os treinamentos e a avaliação ainda não foram iniciados.

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
├── src/              # Código de dados, modelo, treinamento e avaliação
├── notebooks/        # Exploração e inspeção dos dados
├── docs/             # Material do relatório e da apresentação
├── requirements.txt  # Dependências iniciais propostas
└── README.md
```

As pastas possuem apenas arquivos `.gitkeep` para serem registradas no Git.

## Divisão do trabalho

Grupo com 7 integrantes, conforme autorização do professor. Os nomes serão preenchidos após a escolha das funções.

| Função | Responsabilidade | Entrega prevista |
| --- | --- | --- |
| 1 — Base de dados | Obter e explorar a base; conferir classes e quantidades; preparar uma divisão fixa de treino, validação e teste. | Carregamento, divisão salva e descrição da base. |
| 2 — Preparação das imagens | Definir redimensionamento, normalização e pequenas variações para treino; conferir os lotes. | Preparação das imagens e exemplos visualizados. |
| 3 — Modelo e treinamento inicial (reservada ao responsável por este repositório) | Adaptar a saída da ResNet18 e treinar apenas a camada final. | Primeiro experimento, modelo salvo e histórico. |
| 4 — Ajuste fino | Reutilizar a estrutura de treinamento para ajustar também o último bloco da rede. | Segundo experimento e registro das configurações. |
| 5 — Avaliação | Comparar os modelos na mesma base de teste e analisar os erros. | Métricas, gráficos e análise dos resultados. |
| 6 — Relatório e referências | Organizar as contribuições no template IEEE e revisar o texto. | Relatório em PDF, pronto para impressão. |
| 7 — Apresentação e reprodução | Organizar slides e instruções de execução; conferir a reprodução do projeto. | Apresentação e README atualizado. |

Cada integrante deve escrever sobre sua parte e contribuir com a apresentação. Todos devem compreender o fluxo completo para a arguição.

## Próximos passos

- Preencher os nomes dos responsáveis.
- Definir o ambiente de execução e verificar a disponibilidade de GPU.
- Inspecionar a base e estabelecer a divisão dos dados antes dos experimentos.
- Implementar primeiro um fluxo completo e simples.
- Registrar configurações e resultados para o relatório.

## Dependências

O arquivo `requirements.txt` lista as bibliotecas inicialmente previstas. Nenhuma instalação foi realizada e as versões compatíveis deverão ser registradas após a validação do ambiente. As instruções de execução serão adicionadas quando houver código funcional.

## Entregas previstas no enunciado

- Código e relatório digital no Canvas até **25/11/2026**, sem enviar a base de dados.
- Relatório em formato de artigo IEEE de duas colunas, com versões digital e impressa iguais; usar A4 para a versão em Word.
- Agendamento presencial da apresentação até **25/11/2026**, após a entrega do relatório impresso.
- Apresentações até **09/12/2026**.

## Referências iniciais

- [EuroSAT — repositório oficial](https://github.com/phelber/EuroSAT)
- [ResNet18 — documentação do Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [Templates de conferência IEEE](https://www.ieee.org/conferences/publishing/templates)
