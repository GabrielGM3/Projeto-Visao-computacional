import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.treinamento import executar_epoca


class RedePequena(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4), nn.ReLU())
        for parametro in self.base.parameters():
            parametro.requires_grad = False
        self.fc = nn.Linear(4, 2)

    def forward(self, imagens):
        return self.fc(self.base(imagens))


class TestTreinamento(unittest.TestCase):
    def test_atualiza_apenas_camada_final_e_preserva_validacao(self):
        torch.manual_seed(42)
        modelo = RedePequena()
        dados = TensorDataset(torch.randn(5, 4), torch.tensor([0, 1, 0, 1, 0]))
        loader = DataLoader(dados, batch_size=2)
        criterio = nn.CrossEntropyLoss()
        otimizador = torch.optim.Adam(modelo.fc.parameters(), lr=0.01)
        antes = {nome: valor.clone() for nome, valor in modelo.state_dict().items()}

        metricas = executar_epoca(modelo, loader, criterio, "cpu", otimizador)
        self.assertGreater(metricas["perda"], 0)
        self.assertTrue(0 <= metricas["acuracia"] <= 1)
        self.assertFalse(torch.equal(antes["fc.weight"], modelo.fc.weight))
        for nome, valor in modelo.base.state_dict().items():
            self.assertTrue(torch.equal(antes[f"base.{nome}"], valor))

        apos_treino = {nome: valor.clone() for nome, valor in modelo.state_dict().items()}
        resultado = executar_epoca(modelo, loader, criterio, "cpu")
        with torch.no_grad():
            saidas = modelo(dados.tensors[0])
            perda_esperada = criterio(saidas, dados.tensors[1]).item()
            acuracia_esperada = (saidas.argmax(1) == dados.tensors[1]).float().mean().item()
        self.assertAlmostEqual(resultado["perda"], perda_esperada, places=6)
        self.assertAlmostEqual(resultado["acuracia"], acuracia_esperada, places=6)
        for nome, valor in modelo.state_dict().items():
            self.assertTrue(torch.equal(apos_treino[nome], valor))


if __name__ == "__main__":
    unittest.main()
