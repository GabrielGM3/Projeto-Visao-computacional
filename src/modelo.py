import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def criar_modelo(num_classes=10):
    modelo = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Congela as camadas da rede original
    for parametro in modelo.parameters():
        parametro.requires_grad = False

    # A nova camada final fica liberada para treino
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    modelo.eval()
    return modelo


if __name__ == "__main__":
    from notebooks.preprocessamento_py import obter_dataloaders

    torch.manual_seed(42)
    treino_dl, _, _, classes = obter_dataloaders(batch_size=4, num_workers=0)
    modelo = criar_modelo(num_classes=len(classes))
    imagens, _ = next(iter(treino_dl))

    # Confere a saida sem treinar o modelo
    with torch.no_grad():
        saidas = modelo(imagens)

    print("Formato das imagens:", tuple(imagens.shape))
    print("Formato da saida:", tuple(saidas.shape))
    print("Parametros liberados para treino:")
    for nome, parametro in modelo.named_parameters():
        if parametro.requires_grad:
            print(nome)
