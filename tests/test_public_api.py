import unittest
import re
from pathlib import Path

import ordinaryshade as osh


class PublicApiTests(unittest.TestCase):
    def test_version_matches_package_metadata(self):
        metadata = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        version = re.search(r'^version = "([^"]+)"', metadata, re.MULTILINE).group(1)
        self.assertEqual(osh.__version__, version)

    def test_project_does_not_import_ordinarylight(self):
        import sys
        self.assertNotIn("ordinarylight", sys.modules)


if __name__ == "__main__":
    unittest.main()

