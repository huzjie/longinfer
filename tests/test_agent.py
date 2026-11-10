# -*- coding: utf-8 -*-
"""Agent 子系统测试。"""
from __future__ import annotations

import unittest




class TestAgent(unittest.TestCase):
    def _config(self):
        from longinfer.config import Config
        return Config()

    def test_run_completes(self):
        from longinfer.agent.runtime import AgentRuntime
        from longinfer.tools import build_default_registry
        rt = AgentRuntime(self._config(), None, build_default_registry())
        traj = rt.run("测试", resume_from=None)
        self.assertIsNotNone(traj.goal)

    def test_checkpoint_roundtrip(self):
        from longinfer.agent.trajectory import Trajectory, Step
        from longinfer.agent.checkpoint import save_checkpoint, load_checkpoint
        import tempfile, os
        traj = Trajectory(goal="g")
        traj.add(Step(0, thought="t", action="a"))
        with tempfile.TemporaryDirectory() as d:
            p = save_checkpoint(traj, os.path.join(d, "c.json"))
            loaded = load_checkpoint(p)
            self.assertEqual(loaded.goal, "g")
            self.assertEqual(len(loaded.steps), 1)

    def test_rollout_oversampling(self):
        from longinfer.agent.rollout import PartialRollout
        r = PartialRollout(oversample_by_pass_rate=True)
        self.assertEqual(r.adjust_oversampling(1.0), 0)
        self.assertGreaterEqual(r.adjust_oversampling(0.5), 1)

    def test_judge_heuristic(self):
        from longinfer.agent.judge import TrajectoryJudge
        v = TrajectoryJudge(None).evaluate("思考" * 100 + "我认为", "", 5)
        self.assertGreaterEqual(v.score, 0.0)



if __name__ == "__main__":
    unittest.main()
