"""Unit tests for medifed/config.py."""

import unittest
from medifed import config


class TestConfig(unittest.TestCase):

    def test_class_order(self):
        """TC-CFG-001: class order is the fixed global order."""
        self.assertEqual(
            config.CLASS_NAMES,
            ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"],
        )

    def test_class_to_index_consistent(self):
        """TC-CFG-002: CLASS_TO_INDEX matches CLASS_NAMES position for every entry."""
        for idx, name in enumerate(config.CLASS_NAMES):
            self.assertEqual(config.CLASS_TO_INDEX[name], idx)
        self.assertEqual(len(config.CLASS_TO_INDEX), len(config.CLASS_NAMES))

    def test_exactly_three_hospitals(self):
        """TC-CFG-003: HOSPITALS contains exactly 3 entries."""
        self.assertEqual(len(config.HOSPITALS), 3)

    def test_hospital_classes_subset_of_class_names(self):
        """TC-CFG-004: every hospital's class list is a subset of CLASS_NAMES."""
        for hid, info in config.HOSPITALS.items():
            with self.subTest(hospital=hid):
                self.assertTrue(
                    set(info["classes"]).issubset(set(config.CLASS_NAMES)),
                    f"Hospital {hid} has classes outside CLASS_NAMES: {info['classes']}",
                )

    def test_hospital3_classes(self):
        """TC-CFG-005: hospital 3 has COVID19, NORMAL, PNEUMONIA."""
        self.assertEqual(
            sorted(config.HOSPITALS[3]["classes"]),
            ["COVID19", "NORMAL", "PNEUMONIA"],
        )

    def test_hospital2_classes(self):
        """TC-CFG-006: hospital 2 has NORMAL, TUBERCULOSIS."""
        self.assertEqual(
            sorted(config.HOSPITALS[2]["classes"]),
            ["NORMAL", "TUBERCULOSIS"],
        )

    def test_seed_is_int(self):
        """TC-CFG-007: SEED is an integer."""
        self.assertIsInstance(config.SEED, int)

    def test_project_root_config_exists(self):
        """TC-CFG-008: PROJECT_ROOT/medifed/config.py resolves to an existing file."""
        config_path = config.PROJECT_ROOT / "medifed" / "config.py"
        self.assertTrue(config_path.exists(), f"Expected file not found: {config_path}")


if __name__ == "__main__":
    unittest.main()
