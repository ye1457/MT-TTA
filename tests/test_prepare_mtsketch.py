import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_mtsketch as prepare


def record(content=b"original image bytes", name="class_a/sketch.png"):
    return {"task_id": "imagenet_sketch", "relative_path": "sketch/" + name,
            "image_path": "images/imagenet_sketch/" + name, "sample_id": "sample:" + name,
            "file_sha256": hashlib.sha256(content).hexdigest(), "file_bytes": len(content)}


class PrepareSketchTests(unittest.TestCase):
    def test_published_bundle_matches_all_five_runtime_pins(self):
        metadata, rows = prepare.load_metadata(prepare.DEFAULT_METADATA)
        self.assertEqual(set(metadata), set(prepare.FROZEN))
        self.assertEqual(len(rows), 20000)
        self.assertEqual({r["task_id"] for r in rows}, set(prepare.TASKS))

    def test_metadata_archive_rejects_unexpected_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tar.gz"
            with tarfile.open(path, "w:gz") as handle:
                item = tarfile.TarInfo("../unexpected")
                item.size = 1
                handle.addfile(item, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "five frozen files"):
                prepare.load_metadata(path)

    def test_source_paths_reject_traversal_absolute_and_windows_paths(self):
        for value in ("../outside", "/outside", "class/../../outside", "C:/outside", "a\\b", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare.safe_relative(value)

    def test_copy_supports_class_folder_root_without_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "output"
            (source / "class_a").mkdir(parents=True)
            image = source / "class_a/sketch.png"
            content = b"unchanged original bytes"
            image.write_bytes(content)
            item = record(content)
            prepare.copy_from_tree(source, target, [item])
            copied = target / item["image_path"]
            self.assertEqual(copied.read_bytes(), content)
            self.assertNotEqual(copied.stat().st_ino, image.stat().st_ino)
            copied.write_bytes(b"changed copy")
            self.assertEqual(image.read_bytes(), content)

    def test_copy_rejects_same_size_wrong_bytes_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "sketch/class_a").mkdir(parents=True)
            image = source / "sketch/class_a/sketch.png"
            image.write_bytes(b"incorrect")
            with self.assertRaisesRegex(ValueError, "byte-mismatched"):
                prepare.copy_from_tree(source, root / "output", [record(b"expected!")])
            self.assertEqual(image.read_bytes(), b"incorrect")
            self.assertFalse((root / "output").exists())

    def test_copy_rejects_source_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "class_a").mkdir(parents=True)
            outside = root / "outside"
            outside.write_bytes(b"image")
            (source / "class_a/sketch.png").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "outside"):
                prepare.copy_from_tree(source, root / "output", [record(b"image")])

    def test_refuses_existing_output_before_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            keep = output / "keep.txt"
            keep.write_text("keep")
            with self.assertRaisesRegex(ValueError, "new directory"):
                prepare.reconstruct({}, [], {}, output)
            self.assertEqual(keep.read_text(), "keep")

    def test_sketchy_original_and_mirror_layouts_are_supported(self):
        item = {"task_id": "sketchy", "relative_path": "images/dog/1.png",
                "source_relative_path": "256x256/sketch/tx_000000000000/dog/1.png",
                "mirror_archive_member": "sketchy/sketch/tx_000000000000/dog/1.png"}
        paths = prepare.source_candidates(item)
        self.assertIn("dog/1.png", paths)
        self.assertIn("sketch/tx_000000000000/dog/1.png", paths)
        self.assertIn(item["source_relative_path"], paths)
        self.assertIn(item["mirror_archive_member"], paths)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional TU-Berlin Parquet dependency")
    def test_parquet_uses_frozen_row_and_original_bytes(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"preserved embedded bytes"
            table = arrow.table({"image": [{"bytes": b"not selected", "path": "unused.png"},
                                           {"bytes": content, "path": "1.png"}]})
            parquet.write_table(table, root / "test.parquet")
            item = record(content)
            item.update(source_parquet="test.parquet", source_row=1)
            prepare.copy_tuberlin_parquet(root, root / "output", [item])
            self.assertEqual((root / "output" / item["image_path"]).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
