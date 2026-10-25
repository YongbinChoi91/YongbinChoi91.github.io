"""Exercise function extraction through the real Doxygen configuration."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_code_docs as docs


@unittest.skipUnless(shutil.which("doxygen"), "doxygen is required")
class ExtractionTests(unittest.TestCase):
    def test_weak_hooks_and_fresh_xml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            web = root / "Web"
            core = root / "core"
            solver = core / "solver"
            solver.mkdir(parents=True)
            web.mkdir()
            (solver / "hooks.h").write_text(
                "bool supports_model();\n"
                "double model_residual(double value);\n"
                "double model_jvp(double value);\n"
            )
            (solver / "hooks.cpp").write_text(
                '#include "hooks.h"\n'
                "bool __attribute__((weak)) supports_model() { return false; }\n"
                "double __attribute__((weak)) model_residual(double value) { return value; }\n"
                "double __attribute__((weak)) model_jvp(double value) { return value; }\n"
            )
            config = web / "Doxyfile"
            config.write_text(docs.DOXYFILE.read_text())
            xml = web / ".generated/doxygen/xml"
            with patch.multiple(docs, ROOT=root, WEB=web, CORE=core,
                                DOXYFILE=config, XML_DIR=xml):
                docs.run_doxygen()
                functions, registry = docs.parse_functions(docs.core_files())
                expected = {"supports_model", "model_residual", "model_jvp"}
                self.assertEqual({fn.name for fn in registry.values()}, expected)
                self.assertEqual(len(registry), 3)
                for name in ("hooks.cpp", "hooks.h"):
                    self.assertEqual({fn.name for fn in functions["solver"][name]}, expected)
                for fn in registry.values():
                    self.assertIn(fn.name, fn.code)
                    self.assertEqual(fn.file, "core/solver/hooks.cpp")
                    self.assertEqual(fn.declaration_file, "core/solver/hooks.h")
                stale = xml / "removed_source.xml"
                stale.write_text("<doxygen/>")
                docs.run_doxygen()
                self.assertFalse(stale.exists(), "old XML must not survive regeneration")


if __name__ == "__main__":
    unittest.main()
