import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from notebooks.preprocessamento_py import CAMINHO_SPLIT, IMG_SIZE, obter_dataloaders
from src.modelo import criar_modelo
from src.treinamento import executar_epoca

CAMADAS_AJUSTE_FINO = ("layer4", "fc")


def configurar_camadas_ajuste_fino(modelo, camadas=CAMADAS_AJUSTE_FINO):
    """Congela toda a rede e libera apenas as camadas selecionadas para ajuste fino."""
    for parametro in modelo.parameters():
        parametro.requires_grad = False

    for nome_camada in camadas:
        if not hasattr(modelo, nome_camada):
            raise AttributeError(f"O modelo nao possui a camada '{nome_camada}'.")
        for parametro in getattr(modelo, nome_camada).parameters():
            parametro.requires_grad = True

    modelo.eval()
    return modelo


def contar_parametros_treinaveis(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def criar_modelo_ajuste_fino(num_classes=10, caminho_modelo_inicial=None,
                             dispositivo="cpu", classes_esperadas=None):
    """Cria a ResNet18 para ajuste fino (layer4 + fc), opcionalmente partindo da Etapa 3."""
    modelo = criar_modelo(num_classes=num_classes)
    info_origem = {
        "origem_pesos": "IMAGENET1K_V1",
        "caminho_modelo_inicial": None,
        "modelo_inicial_sha256": None,
        "epoca_modelo_inicial": None,
        "acuracia_validacao_inicial": None,
    }

    if caminho_modelo_inicial:
        caminho = Path(caminho_modelo_inicial)
        if not caminho.is_file():
            raise FileNotFoundError(
                f"Checkpoint inicial nao encontrado: {caminho}"
            )

        salvo = torch.load(caminho, map_location=dispositivo, weights_only=True)
        classes_salvas = salvo.get("classes")
        if (
            classes_esperadas is not None
            and classes_salvas is not None
            and list(classes_salvas) != list(classes_esperadas)
        ):
            raise ValueError(
                "As classes do checkpoint inicial nao coincidem com a divisao atual."
            )

        modelo.load_state_dict(salvo["modelo"])
        validacao_inicial = salvo.get("validacao") or {}
        info_origem = {
            "origem_pesos": "treinamento_inicial",
            "caminho_modelo_inicial": str(caminho),
            "modelo_inicial_sha256": hashlib.sha256(caminho.read_bytes()).hexdigest(),
            "epoca_modelo_inicial": salvo.get("epoca"),
            "acuracia_validacao_inicial": validacao_inicial.get("acuracia"),
        }

    configurar_camadas_ajuste_fino(modelo, CAMADAS_AJUSTE_FINO)
    return modelo.to(dispositivo), info_origem


def salvar_grafico_historico(historico, caminho_figura):
    """Salva curvas de perda e acuracia de treino e validacao em PNG."""
    epocas = [item["epoca"] for item in historico]
    perda_treino = [item["treino"]["perda"] for item in historico]
    perda_val = [item["validacao"]["perda"] for item in historico]
    acc_treino = [item["treino"]["acuracia"] * 100 for item in historico]
    acc_val = [item["validacao"]["acuracia"] * 100 for item in historico]

    melhor = max(historico, key=lambda item: item["validacao"]["acuracia"])
    melhor_epoca = melhor["epoca"]
    melhor_acc = melhor["validacao"]["acuracia"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(epocas, perda_treino, marker="o", label="Treino")
    ax1.plot(epocas, perda_val, marker="s", label="Validacao")
    ax1.set_title("Perda por epoca (Ajuste fino)")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("CrossEntropyLoss")
    ax1.set_xticks(epocas)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.plot(epocas, acc_treino, marker="o", label="Treino")
    ax2.plot(epocas, acc_val, marker="s", label="Validacao")
    ax2.axvline(melhor_epoca, color="gray", linestyle=":", alpha=0.8)
    ax2.scatter([melhor_epoca], [melhor_acc], zorder=5, label=f"Melhor ({melhor_acc:.2f}%)")
    ax2.set_title("Acuracia por epoca (Ajuste fino)")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Acuracia (%)")
    ax2.set_xticks(epocas)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    caminho_figura = Path(caminho_figura)
    caminho_figura.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_figura, dpi=150)
    plt.close(fig)


def treinar_ajuste_fino(epocas=5, batch_size=64, taxa_aprendizado=0.0001,
                        num_workers=0, data_dir="./data/eurosat/2750", seed=42,
                        pasta_saida="outputs/ajuste_fino",
                        caminho_modelo_inicial=None,
                        caminho_figura="docs/figuras/ajuste_fino.png",
                        caminho_resultados_docs=None):
    if epocas < 1:
        raise ValueError("O numero de epocas deve ser maior que zero.")
    if taxa_aprendizado <= 0:
        raise ValueError("A taxa de aprendizado deve ser maior que zero.")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    treino_dl, val_dl, _, classes = obter_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    modelo, info_origem = criar_modelo_ajuste_fino(
        num_classes=len(classes),
        caminho_modelo_inicial=caminho_modelo_inicial,
        dispositivo=dispositivo,
        classes_esperadas=classes,
    )

    criterio = nn.CrossEntropyLoss()
    parametros_treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.Adam(parametros_treinaveis, lr=taxa_aprendizado)

    historico = []
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_modelo = pasta_saida / "melhor_modelo.pth"
    melhor_acuracia = -1.0
    melhor_epoca = 0

    configuracao = {
        "etapa": "ajuste_fino",
        "modelo": "resnet18",
        "pesos_iniciais": "IMAGENET1K_V1",
        "origem_pesos": info_origem["origem_pesos"],
        "caminho_modelo_inicial": info_origem["caminho_modelo_inicial"],
        "modelo_inicial_sha256": info_origem["modelo_inicial_sha256"],
        "epoca_modelo_inicial": info_origem["epoca_modelo_inicial"],
        "acuracia_validacao_inicial": info_origem["acuracia_validacao_inicial"],
        "camadas_treinadas": list(CAMADAS_AJUSTE_FINO),
        "parametros_treinaveis": contar_parametros_treinaveis(modelo),
        "parametros_totais": sum(p.numel() for p in modelo.parameters()),
        "epocas": epocas,
        "batch_size": batch_size,
        "taxa_aprendizado": taxa_aprendizado,
        "otimizador": "Adam",
        "criterio": "CrossEntropyLoss",
        "img_size": IMG_SIZE,
        "seed": seed,
        "num_workers": num_workers,
        "dispositivo": str(dispositivo),
        "torch": str(torch.__version__),
        "threads_cpu": torch.get_num_threads(),
        "split": str(CAMINHO_SPLIT),
        "split_sha256": hashlib.sha256(Path(CAMINHO_SPLIT).read_bytes()).hexdigest(),
        "imagens_treino": len(treino_dl.dataset),
        "imagens_validacao": len(val_dl.dataset),
    }

    print(
        f"Dispositivo: {dispositivo} | Origem: {info_origem['origem_pesos']} | "
        f"Parametros treinaveis: {configuracao['parametros_treinaveis']:,}",
        flush=True,
    )

    for epoca in range(1, epocas + 1):
        inicio = time.perf_counter()
        print(f"Epoca {epoca}/{epocas}", flush=True)
        treino = executar_epoca(
            modelo,
            treino_dl,
            criterio,
            dispositivo,
            otimizador,
            base_congelada=False,
            camadas_treino=CAMADAS_AJUSTE_FINO,
        )
        validacao = executar_epoca(modelo, val_dl, criterio, dispositivo)
        historico.append({
            "epoca": epoca,
            "treino": treino,
            "validacao": validacao,
            "tempo_segundos": time.perf_counter() - inicio,
        })

        # Escolhe o melhor modelo usando apenas a validacao (empate mantem a 1a)
        if validacao["acuracia"] > melhor_acuracia:
            melhor_acuracia = validacao["acuracia"]
            melhor_epoca = epoca
            torch.save({
                "modelo": modelo.state_dict(),
                "epoca": epoca,
                "validacao": validacao,
                "classes": classes,
                "configuracao": configuracao,
            }, caminho_modelo)

        resultado = {
            "configuracao": configuracao,
            "classes": classes,
            "melhor_epoca": melhor_epoca,
            "melhor_acuracia_validacao": melhor_acuracia,
            "historico": historico,
        }
        caminho_resultados = pasta_saida / "resultados.json"
        with open(caminho_resultados, "w", encoding="utf-8") as arquivo:
            json.dump(resultado, arquivo, indent=2)

        print(
            f"Treino: perda={treino['perda']:.4f}, acuracia={treino['acuracia']:.2%} | "
            f"Validacao: perda={validacao['perda']:.4f}, "
            f"acuracia={validacao['acuracia']:.2%}",
            flush=True,
        )

    if caminho_figura:
        salvar_grafico_historico(historico, caminho_figura)

    if caminho_resultados_docs:
        destino_docs = Path(caminho_resultados_docs)
        destino_docs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pasta_saida / "resultados.json", destino_docs)

    salvo = torch.load(caminho_modelo, map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(salvo["modelo"])
    modelo.eval()
    print(f"Melhor epoca: {melhor_epoca}. Arquivos salvos em: {pasta_saida}", flush=True)
    return modelo, historico


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ajuste fino da ResNet18 liberando o ultimo bloco (layer4) e a camada final (fc)."
    )
    parser.add_argument("--epocas", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--taxa-aprendizado", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dados", default="./data/eurosat/2750")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saida", default="outputs/ajuste_fino")
    parser.add_argument(
        "--modelo-inicial",
        default=None,
        help="Caminho opcional para o melhor_modelo.pth da etapa 3 (warm start).",
    )
    parser.add_argument("--figura", default="docs/figuras/ajuste_fino.png")
    parser.add_argument(
        "--resultados-docs",
        default="docs/resultados_ajuste_fino.json",
    )
    args = parser.parse_args()

    treinar_ajuste_fino(
        epocas=args.epocas,
        batch_size=args.batch_size,
        taxa_aprendizado=args.taxa_aprendizado,
        num_workers=args.num_workers,
        data_dir=args.dados,
        seed=args.seed,
        pasta_saida=args.saida,
        caminho_modelo_inicial=args.modelo_inicial or None,
        caminho_figura=args.figura or None,
        caminho_resultados_docs=args.resultados_docs or None,
    )