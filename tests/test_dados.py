"""Verificações sem download: divisão oficial e fluxo com imagens temporárias."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import torch
from PIL import Image

from notebooks import dataset_py as dados
from notebooks import preprocessamento_py as prep


ROOT = Path(__file__).resolve().parents[1]


class TestDivisaoOficial(unittest.TestCase):
    def test_cobertura_sem_sobreposicao_e_mapeamento_consistente(self):
        split = json.loads((ROOT / "docs/data_split.json").read_text())
        mapa = json.loads((ROOT / "docs/mapeamento_classes_objetivo.json").read_text())
        conjuntos = [split[k] for k in ("treino", "validacao", "teste")]
        self.assertEqual([len(c) for c in conjuntos], [18900, 4050, 4050])
        todos = [p for c in conjuntos for p in c]
        self.assertEqual(len(set(todos)), 27000)
        self.assertEqual(sorted(split["class_to_idx"].values()), list(range(10)))
        for classe, idx in split["class_to_idx"].items():
            self.assertEqual(mapa[classe]["id_classe"], idx)
            total = mapa[classe]["resumo_quantidades"]["total"]
            for objetivo, ratio in [("treino", .7), ("validacao", .15), ("teste", .15)]:
                paths = [p for p in split[objetivo] if p.split('/')[0] == classe]
                self.assertEqual(paths, mapa[classe]["objetivos"][objetivo])
                self.assertEqual(len(paths), int(total * ratio))
                for path in paths:
                    self.assertFalse(Path(path).is_absolute())
                    self.assertNotIn('..', Path(path).parts)

    def test_importacao_nao_baixa_base(self):
        with patch.object(dados.datasets, "EuroSAT", side_effect=AssertionError("Download no import")):
            importlib.reload(dados)
            importlib.reload(prep)

    def test_download_explicito_retorna_pasta_das_classes(self):
        with patch.object(dados.datasets, "EuroSAT") as download:
            pasta = dados.baixar_dataset("dados-locais")
        download.assert_called_once_with(root="dados-locais", download=True)
        self.assertEqual(Path(pasta), Path("dados-locais/eurosat/2750"))


class TestFluxoImagens(unittest.TestCase):
    def test_divisao_reproduzivel_lotes_e_avaliacao_deterministica(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for classe in ["ClasseA", "ClasseB"]:
                (root / classe).mkdir()
                for i in range(20):
                    Image.new("RGB", (64, 64), (i * 10, 80, 140)).save(root / classe / f"{i}.jpg")
            split_path = str(root / "split.json")
            split = dados.gerar_ou_carregar_split(str(root), split_path)
            repetido = dados.gerar_ou_carregar_split(str(root), split_path, forcar_recriacao=True)
            self.assertEqual(split, repetido)
            loaders = prep.obter_dataloaders(str(root), batch_size=4, num_workers=0, split_json_path=split_path)
            treino, val, teste, classes = loaders
            self.assertEqual([len(dl.dataset) for dl in (treino, val, teste)], [28, 6, 6])
            self.assertEqual(classes, ["ClasseA", "ClasseB"])
            for dl in (treino, val, teste):
                x, y = next(iter(dl))
                self.assertEqual(tuple(x.shape), (4, 3, 224, 224))
                self.assertEqual(x.dtype, torch.float32)
                self.assertEqual(y.dtype, torch.int64)
                self.assertTrue(torch.isfinite(x).all())
                self.assertTrue(((y >= 0) & (y < 2)).all())
            for dl in (val, teste):
                self.assertTrue(torch.equal(dl.dataset[0][0], dl.dataset[0][0]))
            with patch.object(prep.plt, "show"):
                prep.plotar_grade_lote(val, classes, salvar_em=str(root / "grade.png"))
                prep.plotar_aumentos(treino.dataset, 0, salvar_em=str(root / "aumentos.png"))
            self.assertTrue((root / "grade.png").is_file())
            self.assertTrue((root / "aumentos.png").is_file())
            prep.plt.close("all")
            with self.assertRaises(FileNotFoundError):
                prep.obter_dataloaders(str(root), split_json_path=str(root / "inexistente.json"))


if __name__ == "__main__":
    unittest.main()
