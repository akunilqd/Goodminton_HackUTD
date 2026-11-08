"""Behavior tests: python -B -m unittest module4.test_detector -v."""

import copy
import unittest
import json
import tempfile
from pathlib import Path

from config import Settings
from module4 import HitDetector, detect_saved_run


def clip(kind="hit", fps=25, scale=1.):
    rows = []
    for i in range(60):
        if kind == "hit":
            x = 500 - abs(i - 25) * 10
        elif kind == "serve":
            x = 500 + max(0, i - 25) * 10
        else:
            x = 250 + i * 10
        joints = [[0., 0., 0.] for _ in range(17)]
        joints[9] = [500 * scale, 400 * scale, .9]
        rows.append({
            "frame_index": i, "timestamp_s": i / fps,
            "players": [{"track_id": 1, "bbox_xyxy": [400*scale, 200*scale, 600*scale, 600*scale],
                         "keypoints": joints}],
            "shuttle": {"pixel": [x*scale, 400*scale], "raw_pixel": [x*scale, 400*scale],
                        "visible": True, "source": "tracknet"},
        })
    return rows


class HitTests(unittest.TestCase):
    def run_detector(self, rows, **kwargs):
        return HitDetector(Settings()).detect(rows, 25, 1920, 1080, **kwargs)

    def test_one_turn_one_hit(self):
        events, _ = self.run_detector(clip())
        self.assertEqual(len(events), 1)
        self.assertLessEqual(abs(events[0].frame_index - 25), 2)
        self.assertEqual(events[0].hitter_id, 1)

    def test_straight_motion_not_a_hit(self):
        events, _ = self.run_detector(clip("straight"))
        self.assertEqual(events, [])

    def test_serve_from_rest(self):
        events, _ = self.run_detector(clip("serve"))
        self.assertTrue(events)
        self.assertLessEqual(abs(events[0].frame_index - 25), 3)
        self.assertIn("possible_serve", events[0].quality_flags)

    def test_long_gap_not_bridged(self):
        rows = clip()
        for i in range(21, 30):
            rows[i]["shuttle"].update(pixel=None, visible=False, source="missing")
        events, summary = self.run_detector(rows)
        self.assertEqual(events, [])
        self.assertEqual(summary["interpolated_frames"], 0)
        self.assertEqual(summary["unusable_ranges"], [{"start_frame": 21, "end_frame": 29}])

    def test_short_gap_retains_timeline_and_marks_quality(self):
        rows = clip()
        rows[24]["shuttle"].update(pixel=None, visible=False, source="missing")
        events, summary = self.run_detector(rows)
        self.assertEqual(summary["frame_count"], 60)
        self.assertEqual(summary["interpolated_frames"], 1)
        self.assertEqual(len(events), 1)
        self.assertIn("interpolated_shuttle_context", events[0].quality_flags)

    def test_no_player_near_bounce(self):
        rows = clip()
        for row in rows:
            row["players"][0]["keypoints"][9][:2] = [1500, 800]
        self.assertEqual(self.run_detector(rows)[0], [])

    def test_ambiguous_hitter_stays_unknown(self):
        rows = clip()
        for row in rows:
            other = copy.deepcopy(row["players"][0])
            other["track_id"] = 2
            row["players"].append(other)
        events, _ = self.run_detector(rows)
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].hitter_id)
        self.assertIn("ambiguous_hitter", events[0].quality_flags)

    def test_repaired_provenance(self):
        rows = clip()
        for row in rows:
            row["shuttle"]["source"] = "inpaintnet"
        events, _ = self.run_detector(rows)
        self.assertIn("repaired_shuttle_context", events[0].quality_flags)

    def test_resolution_invariance(self):
        first, _ = self.run_detector(clip())
        second, _ = HitDetector(Settings()).detect(clip(scale=.5), 25, 960, 540)
        self.assertEqual(first, second)

    def test_invalid_timeline_and_disabled_modules(self):
        for mutation in (lambda r: r[5].update(frame_index=4),
                         lambda r: r[5].update(timestamp_s=99),
                         lambda r: r[5].update(players=None)):
            rows = clip()
            mutation(rows)
            with self.assertRaises(ValueError):
                self.run_detector(rows)

    def test_rally_boundary_not_crossed(self):
        events, _ = self.run_detector(clip(), rallies=[
            {"rally_id": "a", "start_frame": 0, "end_frame": 24},
            {"rally_id": "b", "start_frame": 25, "end_frame": 59},
        ])
        self.assertEqual(events, [])

    def test_rally_id_and_invalid_overlap(self):
        events, _ = self.run_detector(clip(), rallies=[
            {"rally_id": "a", "start_frame": 0, "end_frame": 59},
        ])
        self.assertEqual(events[0].rally_id, "a")
        self.assertNotIn("rally_unverified", events[0].quality_flags)
        with self.assertRaises(ValueError):
            self.run_detector(clip(), rallies=[
                {"rally_id": 1, "start_frame": 0, "end_frame": 30},
                {"rally_id": 2, "start_frame": 30, "end_frame": 59},
            ])

    def test_missing_wrists_can_preserve_strong_hit_without_hitter(self):
        rows = clip()
        for row in rows:
            row["players"] = []
            row["shuttle"]["pixel"][0] = 700 - abs(row["frame_index"] - 25) * 20
        events, _ = self.run_detector(rows)
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].hitter_id)
        self.assertIn("missing_wrist_evidence", events[0].quality_flags)

    def test_short_clip_no_fabricated_hits(self):
        events, summary = self.run_detector(clip()[:4])
        self.assertEqual(events, [])
        self.assertEqual(summary["evaluated_frames"], 0)

    def test_no_modification_of_inputs(self):
        rows = clip()
        rows[24]["shuttle"].update(pixel=None, visible=False)
        original = copy.deepcopy(rows)
        self.run_detector(rows)
        self.assertEqual(rows, original)

    def test_fps_windows(self):
        for fps in (30, 60):
            rows = clip(fps=fps)
            events, _ = HitDetector(Settings()).detect(rows, fps, 1920, 1080)
            self.assertTrue(events)
            self.assertLessEqual(abs(events[0].timestamp_s - 25/fps), .08)

    def test_saved_replay_preserves_source_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"source", Path(tmp)/"hits"
            source.mkdir()
            rows = clip()
            metadata = {"status": "complete", "modules": {"m1": True, "m2": True},
                        "video": {"fps": 25, "width": 1920, "height": 1080}, "processed_frames": 60}
            (source/"metadata.json").write_text(json.dumps(metadata))
            content = "\n".join(json.dumps(r) for r in rows)
            (source/"frames.jsonl").write_text(content)
            summary = detect_saved_run(Settings(), source, output)
            self.assertEqual(summary["hit_count"], 1)
            self.assertEqual((source/"frames.jsonl").read_text(), content)
            self.assertTrue((output/"hits.jsonl").exists())
            with self.assertRaises(ValueError):
                detect_saved_run(Settings(), source, output)
            metadata["processed_frames"] = 61
            (source/"metadata.json").write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                detect_saved_run(Settings(), source, Path(tmp)/"invalid")
            self.assertFalse((Path(tmp)/"invalid").exists())


if __name__ == "__main__":
    unittest.main()
