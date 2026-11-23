import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ajuste_fino import (
    CAMADAS_AJUSTE_FINO,
    configurar_camadas_ajuste_fino,
    treinar_ajuste_fino,
)
from src.treinamento import executar_epoca


class RedeAjusteFinoPequena(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU())
        self.fc = nn.Linear(4, 2)

    def forward(self, imagens):
        return self.fc(self.layer4(self.base(imagens)))


class TestAjusteFino(unittest.TestCase):
    def test_libera_apenas_layer4_e_fc(self):
        modelo = configurar_camadas_ajuste_fino(RedeAjusteFinoPequena())
        for nome, parametro in modelo.named_parameters():
            if nome.startswith(("layer4.", "fc.")):
                self.assertTrue(parametro.requires_grad, f"{nome} deveria estar liberado")
            else:
                self.assertFalse(parametro.requires_grad, f"{nome} deveria estar congelado")

    def test_atualiza_layer4_e_fc_preservando_base_e_validacao(self):
        torch.manual_seed(42)
        modelo = configurar_camadas_ajuste_fino(RedeAjusteFinoPequena())
        dados = TensorDataset(torch.randn(6, 4), torch.tensor([0, 1, 0, 1, 0, 1]))
        loader = DataLoader(dados, batch_size=2)
        criterio = nn.CrossEntropyLoss()
        params = [p for p in modelo.parameters() if p.requires_grad]
        otimizador = torch.optim.Adam(params, lr=0.01)
        antes = {nome: valor.clone() for nome, valor in modelo.state_dict().items()}

        metricas = executar_epoca(
            modelo,
            loader,
            criterio,
            "cpu",
            otimizador,
            base_congelada=False,
            camadas_treino=CAMADAS_AJUSTE_FINO,
        )
        self.assertGreater(metricas["perda"], 0)
        self.assertTrue(0 <= metricas["acuracia"] <= 1)

        # layer4 e fc devem ser atualizados
        self.assertFalse(torch.equal(antes["layer4.0.weight"], modelo.layer4[0].weight))
        self.assertFalse(torch.equal(antes["fc.weight"], modelo.fc.weight))

        # Base congelada (pesos e estatisticas de BatchNorm) deve ficar intacta
        for nome, valor in modelo.base.state_dict().items():
            self.assertTrue(
                torch.equal(antes[f"base.{nome}"], valor),
                f"base.{nome} nao deveria ter sido alterado",
            )

        # Validacao nao deve alterar nada
        apos_treino = {nome: valor.clone() for nome, valor in modelo.state_dict().items()}
        val = executar_epoca(modelo, loader, criterio, "cpu")
        with torch.no_grad():
            saidas = modelo(dados.tensors[0])
            perda_esperada = criterio(saidas, dados.tensors[1]).item()
            acc_esperada = (saidas.argmax(1) == dados.tensors[1]).float().mean().item()
        self.assertAlmostEqual(val["perda"], perda_esperada, places=6)
        self.assertAlmostEqual(val["acuracia"], acc_esperada, places=6)
        for nome, valor in modelo.state_dict().items():
            self.assertTrue(torch.equal(apos_treino[nome], valor))

    def test_treinar_partindo_do_checkpoint_etapa3(self):
        torch.manual_seed(42)
        modelo_etapa3 = RedeAjusteFinoPequena()
        dados = TensorDataset(torch.randn(6, 4), torch.tensor([0, 1, 0, 1, 0, 1]))
        loader = DataLoader(dados, batch_size=2)
        classes = ["ClasseA", "ClasseB"]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_inicial = tmp_path / "etapa3.pth"
            torch.save({
                "modelo": modelo_etapa3.state_dict(),
                "epoca": 3,
                "validacao": {"perda": 0.21, "acuracia": 0.93},
                "classes": classes,
                "configuracao": {"camadas_treinadas": ["fc"]},
            }, ckpt_inicial)

            figura = tmp_path / "ajuste_fino.png"
            docs_json = tmp_path / "resultados_ajuste_fino.json"
            modelo_novo = RedeAjusteFinoPequena()

            with patch("src.ajuste_fino.criar_modelo", return_value=modelo_novo), patch(
                "src.ajuste_fino.obter_dataloaders", return_value=(loader, loader, None, classes)
            ), patch("torch.cuda.is_available", return_value=False):
                retornado, historico = treinar_ajuste_fino(
                    epocas=2,
                    taxa_aprendizado=0.0001,
                    pasta_saida=tmp_path / "saida",
                    caminho_modelo_inicial=ckpt_inicial,
                    caminho_figura=figura,
                    caminho_resultados_docs=docs_json,
                )

            salvo = torch.load(tmp_path / "saida" / "melhor_modelo.pth", weights_only=True)
            resultado = json.loads((tmp_path / "saida" / "resultados.json").read_text())
            melhor = max(historico, key=lambda item: item["validacao"]["acuracia"])

            self.assertEqual(salvo["epoca"], melhor["epoca"])
            self.assertEqual(resultado["melhor_epoca"], melhor["epoca"])
            self.assertEqual(salvo["classes"], classes)
            self.assertEqual(salvo["configuracao"]["camadas_treinadas"], ["layer4", "fc"])
            self.assertEqual(salvo["configuracao"]["origem_pesos"], "treinamento_inicial")
            self.assertEqual(salvo["configuracao"]["epoca_modelo_inicial"], 3)
            self.assertTrue(figura.is_file())
            self.assertTrue(docs_json.is_file())
            for nome, valor in retornado.state_dict().items():
                self.assertTrue(torch.equal(valor, salvo["modelo"][nome]))

    def test_treinar_partindo_do_imagenet_e_rejeita_classes_divergentes(self):
        torch.manual_seed(42)
        dados = TensorDataset(torch.randn(6, 4), torch.tensor([0, 1, 0, 1, 0, 1]))
        loader = DataLoader(dados, batch_size=2)
        classes = ["ClasseA", "ClasseB"]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch("src.ajuste_fino.criar_modelo", return_value=RedeAjusteFinoPequena()), patch(
                "src.ajuste_fino.obter_dataloaders", return_value=(loader, loader, None, classes)
            ), patch("torch.cuda.is_available", return_value=False):
                _, _ = treinar_ajuste_fino(
                    epocas=1,
                    pasta_saida=tmp_path / "imagenet",
                    caminho_modelo_inicial=None,
                    caminho_figura=None,
                )

            resultado = json.loads((tmp_path / "imagenet" / "resultados.json").read_text())
            self.assertEqual(resultado["configuracao"]["origem_pesos"], "IMAGENET1K_V1")
            self.assertIsNone(resultado["configuracao"]["caminho_modelo_inicial"])

            ckpt_errado = tmp_path / "errado.pth"
            torch.save({
                "modelo": RedeAjusteFinoPequena().state_dict(),
                "epoca": 1,
                "validacao": {"perda": 0.5, "acuracia": 0.5},
                "classes": ["ClasseB", "ClasseA"],
                "configuracao": {},
            }, ckpt_errado)

            with patch("src.ajuste_fino.criar_modelo", return_value=RedeAjusteFinoPequena()), patch(
                "src.ajuste_fino.obter_dataloaders", return_value=(loader, loader, None, classes)
            ), patch("torch.cuda.is_available", return_value=False):
                with self.assertRaises(ValueError):
                    treinar_ajuste_fino(
                        epocas=1,
                        pasta_saida=tmp_path / "falha",
                        caminho_modelo_inicial=ckpt_errado,
                        caminho_figura=None,
                    )


if __name__ == "__main__":
    unittest.main()