import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from notebooks.preprocessamento_py import CAMINHO_SPLIT, IMG_SIZE, obter_dataloaders
from src.modelo import criar_modelo


def executar_epoca(modelo, dataloader, criterio, dispositivo, otimizador=None,
                   base_congelada=True, camadas_treino=None):
    treino = otimizador is not None
    modelo.train(treino)

    # Mantem as estatisticas das camadas congeladas
    if camadas_treino is not None:
        modelo.eval()
        for nome_camada in camadas_treino:
            getattr(modelo, nome_camada).train(treino)
    elif base_congelada:
        modelo.eval()
        modelo.fc.train(treino)

    perda_total = 0.0
    acertos = 0
    total = 0

    for lote, (imagens, rotulos) in enumerate(dataloader, start=1):
        imagens = imagens.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        if treino:
            otimizador.zero_grad()

        with torch.set_grad_enabled(treino):
            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)

            if treino:
                perda.backward()
                otimizador.step()

        perda_total += perda.item() * rotulos.size(0)
        acertos += (saidas.argmax(dim=1) == rotulos).sum().item()
        total += rotulos.size(0)

        if lote % 50 == 0:
            etapa = "Treino" if treino else "Validacao"
            print(f"  {etapa}: lote {lote}/{len(dataloader)}", flush=True)

    return {"perda": perda_total / total, "acuracia": acertos / total}


def treinar(epocas=3, batch_size=64, taxa_aprendizado=0.001, num_workers=0,
            data_dir="./data/eurosat/2750", seed=42,
            pasta_saida="outputs/treinamento_inicial"):
    if epocas < 1:
        raise ValueError("O numero de epocas deve ser maior que zero.")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    treino_dl, val_dl, _, classes = obter_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    modelo = criar_modelo(num_classes=len(classes)).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.fc.parameters(), lr=taxa_aprendizado)
    historico = []
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_modelo = pasta_saida / "melhor_modelo.pth"
    melhor_acuracia = -1.0
    melhor_epoca = 0
    configuracao = {
        "modelo": "resnet18",
        "pesos_iniciais": "IMAGENET1K_V1",
        "camadas_treinadas": ["fc"],
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
        "split": CAMINHO_SPLIT,
        "split_sha256": hashlib.sha256(Path(CAMINHO_SPLIT).read_bytes()).hexdigest(),
        "imagens_treino": len(treino_dl.dataset),
        "imagens_validacao": len(val_dl.dataset),
    }

    print(f"Dispositivo: {dispositivo}", flush=True)
    for epoca in range(1, epocas + 1):
        inicio = time.perf_counter()
        print(f"Epoca {epoca}/{epocas}", flush=True)
        treino = executar_epoca(modelo, treino_dl, criterio, dispositivo, otimizador)
        validacao = executar_epoca(modelo, val_dl, criterio, dispositivo)
        historico.append({
            "epoca": epoca,
            "treino": treino,
            "validacao": validacao,
            "tempo_segundos": time.perf_counter() - inicio,
        })

        # Escolhe o melhor modelo usando apenas a validacao
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
        with open(pasta_saida / "resultados.json", "w", encoding="utf-8") as arquivo:
            json.dump(resultado, arquivo, indent=2)

        print(
            f"Treino: perda={treino['perda']:.4f}, acuracia={treino['acuracia']:.2%} | "
            f"Validacao: perda={validacao['perda']:.4f}, "
            f"acuracia={validacao['acuracia']:.2%}",
            flush=True,
        )

    salvo = torch.load(caminho_modelo, map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(salvo["modelo"])
    modelo.eval()
    print(f"Melhor epoca: {melhor_epoca}. Arquivos salvos em: {pasta_saida}", flush=True)
    return modelo, historico


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina a camada final da ResNet18.")
    parser.add_argument("--epocas", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--taxa-aprendizado", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dados", default="./data/eurosat/2750")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saida", default="outputs/treinamento_inicial")
    args = parser.parse_args()

    treinar(
        epocas=args.epocas,
        batch_size=args.batch_size,
        taxa_aprendizado=args.taxa_aprendizado,
        num_workers=args.num_workers,
        data_dir=args.dados,
        seed=args.seed,
        pasta_saida=args.saida,
    )
