import unittest

import ordinaryshade as osh


class PublicApiTests(unittest.TestCase):
    def test_version_matches_initial_alpha(self):
        self.assertEqual(osh.__version__, "0.1.0a0")

    def test_project_does_not_import_ordinarylight(self):
        import sys
        self.assertNotIn("ordinarylight", sys.modules)


if __name__ == "__main__":
    unittest.main()

