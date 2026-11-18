import importlib
import pkgutil
import unittest

import pybindlibs


class ImportAllModulesTest(unittest.TestCase):
    def test_every_simulator_module_imports(self):
        names = sorted(
            module.name
            for module in pkgutil.iter_modules(pybindlibs.__path__)
            if module.name.startswith("cpp_")
        )
        self.assertTrue(names)
        for name in names:
            with self.subTest(module=name):
                importlib.import_module(f"pybindlibs.{name}")


if __name__ == "__main__":
    unittest.main()
