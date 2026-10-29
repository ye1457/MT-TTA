"""CPU preflight CLI contracts; hashing must never start model evaluation."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from clip_officehome import evaluation, imagenet_ood


class DataPreflightTests(unittest.TestCase):
    def test_pacs_and_officehome_hashing_is_opt_in_for_dry_run(self):
        for benchmark in ("pacs", "officehome"):
            for hash_images in (False, True):
                with self.subTest(benchmark=benchmark, hash_images=hash_images):
                    output = io.StringIO()

                    def prepare(config, root, hash_images=False):
                        return {}, {}, {"image_bytes_hashed": hash_images}

                    argv = ["--benchmark", benchmark, "--data-root", "/unused/data", "--dry-run"]
                    if hash_images:
                        argv.append("--hash-images")
                    with patch.object(evaluation, "prepare_data", side_effect=prepare) as scan, \
                            patch.object(evaluation, "run", side_effect=AssertionError("Model evaluation called")), \
                            contextlib.redirect_stdout(output):
                        evaluation.main(argv)
                    scan.assert_called_once()
                    self.assertEqual(scan.call_args.kwargs["hash_images"], hash_images)
                    self.assertEqual(scan.call_args.args[1], Path("/unused/data"))
                    result = json.loads(output.getvalue())
                    self.assertEqual(result["status"], "DRY_RUN_PASS_NO_MODEL")
                    self.assertEqual(result["data"]["image_bytes_hashed"], hash_images)

    def test_imagenet_hashing_is_opt_in_for_dry_run(self):
        for benchmark in imagenet_ood.COUNTS:
            for hash_images in (False, True):
                with self.subTest(benchmark=benchmark, hash_images=hash_images):
                    output = io.StringIO()

                    def prepare(name, root, hash_images=False):
                        return {}, {}, {}, {"image_bytes_hashed": hash_images}

                    argv = ["--benchmark", benchmark, "--data-root", "/unused/data", "--dry-run"]
                    if hash_images:
                        argv.append("--hash-images")
                    with patch.object(imagenet_ood, "prepare_imagenet_data", side_effect=prepare) as scan, \
                            patch.object(evaluation, "run_prepared", side_effect=AssertionError("Model evaluation called")), \
                            contextlib.redirect_stdout(output):
                        imagenet_ood.main(argv)
                    scan.assert_called_once_with(benchmark, Path("/unused/data"), hash_images)
                    result = json.loads(output.getvalue())
                    self.assertEqual(result["status"], "DRY_RUN_PASS_NO_MODEL")
                    self.assertEqual(result["data"]["image_bytes_hashed"], hash_images)

    def test_hash_images_without_dry_run_fails_before_data_or_model_access(self):
        for module, scanner, benchmark in (
                (evaluation, "prepare_data", "pacs"),
                (imagenet_ood, "prepare_imagenet_data", "imagenet_a")):
            with self.subTest(benchmark=benchmark):
                errors = io.StringIO()
                with patch.object(module, scanner) as scan, \
                        patch.object(evaluation, "run") as run, \
                        patch.object(evaluation, "run_prepared") as run_prepared, \
                        contextlib.redirect_stderr(errors), \
                        self.assertRaises(SystemExit) as caught:
                    module.main(["--benchmark", benchmark, "--data-root", "/unused/data", "--hash-images"])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("only valid with --dry-run", errors.getvalue())
                scan.assert_not_called()
                run.assert_not_called()
                run_prepared.assert_not_called()

    def test_formal_imagenet_still_hashes_without_the_preflight_flag(self):
        output = io.StringIO()
        with patch.object(imagenet_ood, "prepare_imagenet_data", return_value=({}, {}, {}, {})) as scan, \
                patch.object(evaluation, "run_prepared", return_value={"status": "fixture"}) as run, \
                contextlib.redirect_stdout(output):
            imagenet_ood.main(["--benchmark", "imagenet_a", "--data-root", "/unused/data",
                               "--output-dir", "/unused/output"])
        scan.assert_called_once_with("imagenet_a", Path("/unused/data"), True)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
