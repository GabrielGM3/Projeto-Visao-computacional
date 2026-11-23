import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from notebooks.preprocessamento_py import (
    CAMINHO_DADOS,
    CAMINHO_SPLIT,
    IMG_SIZE,
    desnormalizar,
    obter_dataloaders,
)
from src.modelo import criar_modelo


def carregar_checkpoint(caminho, dispositivo="cpu", classes_esperadas=None):
    """Carrega um checkpoint .pth da Etapa 3 ou 4 em modo de avaliacao."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {caminho}")

    salvo = torch.load(caminho, map_location=dispositivo, weights_only=True)
    classes_salvas = salvo.get("classes")
    if (
        classes_esperadas is not None
        and classes_salvas is not None
        and list(classes_salvas) != list(classes_esperadas)
    ):
        raise ValueError(
            f"As classes de '{caminho}' nao coincidem com o split oficial."
        )

    num_classes = len(classes_esperadas or classes_salvas)
    modelo = criar_modelo(num_classes=num_classes)
    modelo.load_state_dict(salvo["modelo"])
    modelo.to(dispositivo)
    modelo.eval()
    return modelo, salvo


def avaliar_modelo(modelo, dataloader, classes, dispositivo="cpu"):
    """Executa inferencia sem gradientes e calcula metricas, matriz e erros."""
    modelo.eval()
    y_true = []
    y_pred = []
    confiancas = []

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(dispositivo)
            saidas = modelo(imagens)
            probs = torch.softmax(saidas, dim=1)
            confs, preds = probs.max(dim=1)

            y_true.extend(rotulos.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            confiancas.extend(confs.cpu().tolist())

    indices_classes = list(range(len(classes)))
    acuracia = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    matriz = confusion_matrix(y_true, y_pred, labels=indices_classes).tolist()
    relatorio = classification_report(
        y_true,
        y_pred,
        labels=indices_classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    samples = getattr(dataloader.dataset, "samples", None)
    erros = []
    for idx, (real, prev, conf) in enumerate(zip(y_true, y_pred, confiancas)):
        if real != prev:
            arquivo = Path(samples[idx][0]).name if samples else f"amostra_{idx}"
            erros.append({
                "indice": idx,
                "arquivo": arquivo,
                "real_idx": real,
                "real": classes[real],
                "previsto_idx": prev,
                "previsto": classes[prev],
                "confianca": round(float(conf), 4),
            })

    # Ordena pelos erros em que o modelo errou com maior confianca
    erros.sort(key=lambda item: item["confianca"], reverse=True)

    return {
        "acuracia": acuracia,
        "f1_macro": f1_macro,
        "total_imagens": len(y_true),
        "total_acertos": int(sum(r == p for r, p in zip(y_true, y_pred))),
        "total_erros": len(erros),
        "matriz_confusao": matriz,
        "metricas_por_classe": {
            classe: {
                "precisao": float(relatorio[classe]["precision"]),
                "revocacao": float(relatorio[classe]["recall"]),
                "f1": float(relatorio[classe]["f1-score"]),
                "suporte": int(relatorio[classe]["support"]),
            }
            for classe in classes
        },
        "erros": erros,
        "_y_true": y_true,
        "_y_pred": y_pred,
        "_confiancas": confiancas,
    }


def salvar_grafico_comparacao(res_inicial, res_fino, classes, caminho_figura):
    """Plota comparacao geral (Acuracia e F1 Macro) e F1 por classe."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 2.2]}
    )

    # 1) Metricas globais
    metricas = ["Acuracia", "F1 Macro"]
    vals_ini = [res_inicial["acuracia"] * 100, res_inicial["f1_macro"] * 100]
    vals_fin = [res_fino["acuracia"] * 100, res_fino["f1_macro"] * 100]
    x = np.arange(len(metricas))
    largura = 0.34

    b1 = ax1.bar(x - largura / 2, vals_ini, largura, label="Etapa 3 (Camada final)")
    b2 = ax1.bar(x + largura / 2, vals_fin, largura, label="Etapa 4 (Ajuste fino)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metricas)
    ax1.set_ylabel("Pontuacao (%)")
    ax1.set_ylim(80, 101)
    ax1.set_title("Metricas globais no teste")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")
    for barra in list(b1) + list(b2):
        h = barra.get_height()
        ax1.annotate(
            f"{h:.2f}%",
            (barra.get_x() + barra.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=8.5,
            xytext=(0, 3),
            textcoords="offset points",
        )

    # 2) F1 por classe
    xc = np.arange(len(classes))
    f1_ini = [res_inicial["metricas_por_classe"][c]["f1"] * 100 for c in classes]
    f1_fin = [res_fino["metricas_por_classe"][c]["f1"] * 100 for c in classes]

    ax2.bar(xc - largura / 2, f1_ini, largura, label="Etapa 3 (Camada final)")
    ax2.bar(xc + largura / 2, f1_fin, largura, label="Etapa 4 (Ajuste fino)")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(classes, rotation=35, ha="right", fontsize=8.5)
    ax2.set_ylabel("F1-score (%)")
    ax2.set_ylim(75, 101)
    ax2.set_title("F1-score por classe no teste")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    caminho_figura = Path(caminho_figura)
    caminho_figura.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_figura, dpi=150, bbox_inches="tight")
    plt.close(fig)


def salvar_matrizes_confusao(res_inicial, res_fino, classes, caminho_figura):
    """Salva as matrizes de confusao dos dois modelos lado a lado."""
    fig, eixos = plt.subplots(1, 2, figsize=(15, 6.5))
    paineis = [
        (eixos[0], np.array(res_inicial["matriz_confusao"]), "Etapa 3 - Treinamento inicial (fc)"),
        (eixos[1], np.array(res_fino["matriz_confusao"]), "Etapa 4 - Ajuste fino (layer4 + fc)"),
    ]

    for eixo, matriz, titulo in paineis:
        im = eixo.imshow(matriz, cmap="Blues")
        eixo.set_title(titulo, fontsize=11)
        eixo.set_xlabel("Classe prevista")
        eixo.set_ylabel("Classe real")
        eixo.set_xticks(range(len(classes)))
        eixo.set_yticks(range(len(classes)))
        eixo.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        eixo.set_yticklabels(classes, fontsize=8)

        limite = matriz.max() / 2.0 if matriz.size else 0
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                cor = "white" if matriz[i, j] > limite else "black"
                eixo.text(j, i, str(matriz[i, j]), ha="center", va="center", color=cor, fontsize=8)
        fig.colorbar(im, ax=eixo, fraction=0.046, pad=0.04)

    fig.tight_layout()
    caminho_figura = Path(caminho_figura)
    caminho_figura.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_figura, dpi=150, bbox_inches="tight")
    plt.close(fig)


def salvar_grade_erros(dataset_teste, erros, titulo, caminho_figura, max_imagens=9):
    """Salva uma grade com exemplos de erros (imagens desnormalizadas)."""
    selecionados = erros[:max_imagens]
    if not selecionados:
        return

    n_colunas = min(3, len(selecionados))
    n_linhas = (len(selecionados) + n_colunas - 1) // n_colunas
    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(4 * n_colunas, 4 * n_linhas))
    eixos = np.atleast_1d(eixos).flatten()

    for i, eixo in enumerate(eixos):
        eixo.axis("off")
        if i < len(selecionados):
            erro = selecionados[i]
            img_tensor, _ = dataset_teste[erro["indice"]]
            img_vis = desnormalizar(img_tensor).permute(1, 2, 0).numpy()
            eixo.imshow(img_vis)
            eixo.set_title(
                f"Real: {erro['real']}\nPrev: {erro['previsto']} ({erro['confianca']:.1%})",
                fontsize=9.5,
            )

    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    caminho_figura = Path(caminho_figura)
    caminho_figura.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_figura, dpi=150, bbox_inches="tight")
    plt.close(fig)


def avaliar_experimentos(
    caminho_inicial="outputs/treinamento_inicial/melhor_modelo.pth",
    caminho_ajuste_fino="outputs/ajuste_fino/melhor_modelo.pth",
    data_dir=CAMINHO_DADOS,
    batch_size=64,
    num_workers=0,
    pasta_figuras="docs/figuras",
    caminho_resultados_docs="docs/resultados_avaliacao.json",
):
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, teste_dl, classes = obter_dataloaders(
        data_dir=data_dir,
        img_size=IMG_SIZE,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Dispositivo: {dispositivo} | Base de teste: {len(teste_dl.dataset)} imagens", flush=True)

    modelo_ini, ckpt_ini = carregar_checkpoint(caminho_inicial, dispositivo, classes)
    print("Avaliando Etapa 3 (Treinamento inicial)...", flush=True)
    res_ini = avaliar_modelo(modelo_ini, teste_dl, classes, dispositivo)

    modelo_fino, ckpt_fino = carregar_checkpoint(caminho_ajuste_fino, dispositivo, classes)
    print("Avaliando Etapa 4 (Ajuste fino)...", flush=True)
    res_fino = avaliar_modelo(modelo_fino, teste_dl, classes, dispositivo)

    pasta_figuras = Path(pasta_figuras)
    salvar_grafico_comparacao(
        res_ini, res_fino, classes, pasta_figuras / "comparacao_metricas.png"
    )
    salvar_matrizes_confusao(
        res_ini, res_fino, classes, pasta_figuras / "matrizes_confusao.png"
    )
    salvar_grade_erros(
        teste_dl.dataset,
        res_ini["erros"],
        "Exemplos de erros - Etapa 3 (Camada final)",
        pasta_figuras / "exemplos_erros_inicial.png",
    )
    salvar_grade_erros(
        teste_dl.dataset,
        res_fino["erros"],
        "Exemplos de erros - Etapa 4 (Ajuste fino)",
        pasta_figuras / "exemplos_erros_ajuste_fino.png",
    )

    # Comparacao pareada: quantos erros da Etapa 3 foram corrigidos na Etapa 4
    erros_ini_set = {e["indice"] for e in res_ini["erros"]}
    erros_fino_set = {e["indice"] for e in res_fino["erros"]}
    corrigidos = sorted(erros_ini_set - erros_fino_set)
    novos_erros = sorted(erros_fino_set - erros_ini_set)
    erros_comuns = sorted(erros_ini_set & erros_fino_set)

    # Limpa campos internos antes de salvar JSON
    def limpar(res, ckpt, caminho_ckpt):
        copia = {k: v for k, v in res.items() if not k.startswith("_")}
        copia["epoca_checkpoint"] = ckpt.get("epoca")
        copia["checkpoint_sha256"] = hashlib.sha256(Path(caminho_ckpt).read_bytes()).hexdigest()
        copia["top_erros"] = copia.pop("erros")[:20]
        return copia

    resultado_final = {
        "configuracao": {
            "etapa": "avaliacao",
            "split": str(CAMINHO_SPLIT),
            "split_sha256": hashlib.sha256(Path(CAMINHO_SPLIT).read_bytes()).hexdigest(),
            "imagens_teste": len(teste_dl.dataset),
            "img_size": IMG_SIZE,
            "batch_size": batch_size,
            "dispositivo": str(dispositivo),
            "torch": str(torch.__version__),
        },
        "classes": classes,
        "treinamento_inicial": limpar(res_ini, ckpt_ini, caminho_inicial),
        "ajuste_fino": limpar(res_fino, ckpt_fino, caminho_ajuste_fino),
        "comparacao": {
            "ganho_acuracia_pp": round((res_fino["acuracia"] - res_ini["acuracia"]) * 100, 4),
            "ganho_f1_macro_pp": round((res_fino["f1_macro"] - res_ini["f1_macro"]) * 100, 4),
            "reducao_relativa_erros_pct": round(
                (1 - res_fino["total_erros"] / res_ini["total_erros"]) * 100, 2
            ) if res_ini["total_erros"] > 0 else 0.0,
            "erros_corrigidos_pelo_ajuste_fino": len(corrigidos),
            "novos_erros_no_ajuste_fino": len(novos_erros),
            "erros_em_ambos": len(erros_comuns),
        },
    }

    if caminho_resultados_docs:
        destino = Path(caminho_resultados_docs)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(resultado_final, indent=2), encoding="utf-8")

    print("\n=== RESUMO NO CONJUNTO DE TESTE ===")
    print(
        f"Etapa 3 (Inicial)    -> Acuracia: {res_ini['acuracia']:.2%} | "
        f"F1 Macro: {res_ini['f1_macro']:.2%} | Erros: {res_ini['total_erros']}/{res_ini['total_imagens']}"
    )
    print(
        f"Etapa 4 (Ajuste fino)-> Acuracia: {res_fino['acuracia']:.2%} | "
        f"F1 Macro: {res_fino['f1_macro']:.2%} | Erros: {res_fino['total_erros']}/{res_fino['total_imagens']}"
    )
    print(f"Resultados salvos em: {caminho_resultados_docs}")
    print(f"Figuras salvas em: {pasta_figuras}")
    return resultado_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avalia e compara os modelos das Etapas 3 e 4 na base de teste."
    )
    parser.add_argument(
        "--modelo-inicial", default="outputs/treinamento_inicial/melhor_modelo.pth"
    )
    parser.add_argument(
        "--modelo-ajuste-fino", default="outputs/ajuste_fino/melhor_modelo.pth"
    )
    parser.add_argument("--dados", default=CAMINHO_DADOS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--figuras", default="docs/figuras")
    parser.add_argument("--resultados-docs", default="docs/resultados_avaliacao.json")
    args = parser.parse_args()

    avaliar_experimentos(
        caminho_inicial=args.modelo_inicial,
        caminho_ajuste_fino=args.modelo_ajuste_fino,
        data_dir=args.dados,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pasta_figuras=args.figuras,
        caminho_resultados_docs=args.resultados_docs,
    )