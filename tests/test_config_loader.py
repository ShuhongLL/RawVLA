import tempfile
import unittest
from pathlib import Path

from starVLA.config_loader import load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_relative_base_is_merged_and_child_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base.yaml").write_text("model:\n  width: 32\n  depth: 4\n")
            child_dir = root / "children"
            child_dir.mkdir()
            (child_dir / "model.yaml").write_text(
                "extends: ../base.yaml\nmodel:\n  depth: 8\nrun_id: child\n"
            )

            cfg = load_config(child_dir / "model.yaml")

            self.assertEqual(cfg.model.width, 32)
            self.assertEqual(cfg.model.depth, 8)
            self.assertEqual(cfg.run_id, "child")
            self.assertNotIn("extends", cfg)

    def test_multiple_bases_merge_left_to_right(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "first.yaml").write_text("value: 1\nnested:\n  first: true\n")
            (root / "second.yaml").write_text("value: 2\nnested:\n  second: true\n")
            (root / "child.yaml").write_text("extends: [first.yaml, second.yaml]\n")

            cfg = load_config(root / "child.yaml")

            self.assertEqual(cfg.value, 2)
            self.assertTrue(cfg.nested.first)
            self.assertTrue(cfg.nested.second)

    def test_cycle_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.yaml").write_text("extends: b.yaml\n")
            (root / "b.yaml").write_text("extends: a.yaml\n")

            with self.assertRaisesRegex(ValueError, "inheritance cycle"):
                load_config(root / "a.yaml")


if __name__ == "__main__":
    unittest.main()
