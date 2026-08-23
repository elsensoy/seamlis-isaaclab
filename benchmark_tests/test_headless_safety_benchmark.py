from pathlib import Path
import unittest

import yaml

from examples.benchmark_config_suite import classify_outcome
from scripts.utils import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPO_ROOT / "configs" / "benchmarks" / "headless_safety" / "suite.yaml"


class HeadlessSafetyBenchmarkConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))

    def test_all_scenarios_validate_and_are_headless(self):
        for relative_path in self.suite["scenarios"]:
            with self.subTest(relative_path=relative_path):
                cfg = load_config(str(SUITE_PATH.parent / relative_path))
                self.assertFalse(cfg.show_animation)
                self.assertEqual(cfg.dt, 0.1)
                self.assertEqual(cfg.environment.width, 24.0)
                self.assertEqual(cfg.environment.height, 18.0)
                self.assertGreater(len(cfg.robots.instances), 0)
                self.assertLessEqual(len(cfg.robots.instances), 3)

    def test_shared_robot_limits(self):
        for relative_path in self.suite["scenarios"]:
            with self.subTest(relative_path=relative_path):
                common = load_config(str(SUITE_PATH.parent / relative_path)).robots.common
                self.assertEqual(common.radius, 0.15)
                self.assertEqual(common.v_max, 1.5)
                self.assertEqual(common.a_max, 2.0)
                self.assertEqual(common.cam_range, 4.5)
                self.assertEqual(common.mpc_horizon, 10)
                self.assertEqual(common.mpc_cbf_alpha1, 0.55)
                self.assertEqual(common.mpc_cbf_alpha2, 0.55)

    def test_collision_is_not_conflated_with_infeasibility(self):
        collision = classify_outcome(False, "collision_or_infeasible", {"type": "unknown"})
        infeasible = classify_outcome(False, "collision_or_infeasible", {"type": "known_or_infeasible"})
        timeout = classify_outcome(False, "max_steps", None)

        self.assertTrue(collision["collision"])
        self.assertFalse(collision["infeasible"])
        self.assertFalse(infeasible["collision"])
        self.assertTrue(infeasible["infeasible"])
        self.assertTrue(timeout["timeout"])


if __name__ == "__main__":
    unittest.main()
