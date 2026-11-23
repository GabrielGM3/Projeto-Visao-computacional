import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.avaliacao import avaliar_experimentos, avaliar_modelo, carregar_checkpoint


class RedeSimples(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))

    def forward(self, x):
        return self.net(x)


class TestAvaliacao(unittest.TestCase):
    def test_avaliacao_completa_gera_json_e_figuras(self):
        torch.manual_seed(42)
        classes = ["ClasseA", "ClasseB"]
        imagens = torch.randn(6, 3, 8, 8)
        rotulos = torch.tensor([0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(imagens, rotulos), batch_size=3, shuffle=False)

        modelo_a = RedeSimples(2)
        modelo_b = RedeSimples(2)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_ini = tmp_path / "ini.pth"
            ckpt_fino = tmp_path / "fino.pth"
            torch.save({"modelo": modelo_a.state_dict(), "epoca": 3, "classes": classes}, ckpt_ini)
            torch.save({"modelo": modelo_b.state_dict(), "epoca": 4, "classes": classes}, ckpt_fino)

            pasta_fig = tmp_path / "figuras"
            json_saida = tmp_path / "resultados_avaliacao.json"

            with patch("src.avaliacao.criar_modelo", side_effect=[RedeSimples(2), RedeSimples(2)]), patch(
                "src.avaliacao.obter_dataloaders", return_value=(None, None, loader, classes)
            ), patch("torch.cuda.is_available", return_value=False):
                resultado = avaliar_experimentos(
                    caminho_inicial=ckpt_ini,
                    caminho_ajuste_fino=ckpt_fino,
                    pasta_figuras=pasta_fig,
                    caminho_resultados_docs=json_saida,
                )

            self.assertTrue(json_saida.is_file())
            self.assertTrue((pasta_fig / "comparacao_metricas.png").is_file())
            self.assertTrue((pasta_fig / "matrizes_confusao.png").is_file())
            salvo = json.loads(json_saida.read_text(encoding="utf-8"))
            self.assertEqual(salvo["classes"], classes)
            self.assertEqual(salvo["treinamento_inicial"]["total_imagens"], 6)
            self.assertEqual(salvo["ajuste_fino"]["total_imagens"], 6)
            self.assertIn("f1_macro", resultado["ajuste_fino"])

    def test_rejeita_classes_divergentes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "errado.pth"
            torch.save(
                {"modelo": RedeSimples(2).state_dict(), "classes": ["B", "A"]},
                ckpt,
            )
            with patch("src.avaliacao.criar_modelo", return_value=RedeSimples(2)):
                with self.assertRaises(ValueError):
                    carregar_checkpoint(ckpt, "cpu", classes_esperadas=["A", "B"])


if __name__ == "__main__":
    unittest.main()