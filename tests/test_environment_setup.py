from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from main import ensure_isolated_environment


class EnvironmentSetupTests(unittest.TestCase):
    def test_creates_isolated_structure_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)

            paths = ensure_isolated_environment(base_dir)

            self.assertTrue(paths["assistant_dir"].exists())
            self.assertTrue(paths["memory_dir"].exists())
            self.assertTrue(paths["agent_prompt_path"].exists())

            config = json.loads(paths["config_path"].read_text(encoding="utf-8"))
            self.assertTrue(config["plugins"]["opencode-mem"]["enabled"])


if __name__ == "__main__":
    unittest.main()
