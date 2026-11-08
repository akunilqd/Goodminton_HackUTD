"""Behavior tests: python -B -m unittest module7_summary.test_rallies -v."""

import unittest

from module7_summary.rallies import build_rallies, classify_landing

FPS = 25


def frame(i, shuttle=None):
    return {"frame_index": i, "timestamp_s": i / FPS, "players": [],
            "shuttle": {"visible": shuttle is not None, "court_projection_m": shuttle}}


def hit(event, i, who):
    return {"event_id": event, "frame_index": i, "timestamp_s": i / FPS, "hitter_id": who}


def shot(i, side):
    return {"start_frame": i, "player_side": side}


def clip(landing, last=100, length=200):
    """Far player (1) hits last; the shuttle rests at `landing` from frame last+20."""
    rows = [frame(i) for i in range(length)]
    if landing is not None:
        for i in range(last + 20, min(last + 30, length)):
            rows[i] = frame(i, list(landing))
    hits = [hit(1, last - 50, 2), hit(2, last, 1)]
    return rows, hits, [shot(last - 50, "Bottom"), shot(last, "Top")]


class Landing(unittest.TestCase):
    def test_far_hitter(self):
        self.assertEqual(classify_landing((3, 10), "far"), "winner")
        self.assertEqual(classify_landing((3, 14), "far"), "out")
        self.assertEqual(classify_landing((0.1, 10), "far"), "out")
        self.assertEqual(classify_landing((3, 6.7), "far"), "net")
        self.assertEqual(classify_landing((3, 3), "far"), "net")

    def test_near_hitter_mirrors(self):
        self.assertEqual(classify_landing((3, 3), "near"), "winner")
        self.assertEqual(classify_landing((3, 10), "near"), "net")
        self.assertEqual(classify_landing((3, -1), "near"), "out")

    def test_on_the_line_is_not_called(self):
        self.assertEqual(classify_landing((0.46, 10), "far"), "unknown")
        self.assertEqual(classify_landing((3, 13.4), "far"), "unknown")


class Rallies(unittest.TestCase):
    def outcome(self, landing, **kw):
        rallies, _ = build_rallies(*clip(landing, **kw), FPS)
        self.assertEqual(len(rallies), 1)
        return rallies[0]

    def test_in_court_gives_hitter_the_point(self):
        r = self.outcome((3, 10))
        self.assertEqual((r["outcome"], r["winner_id"]), ("winner", 1))
        self.assertEqual(r["score_after"], {"1": 1, "2": 0})
        self.assertTrue(r["score_reliable"])

    def test_out_gives_opponent_the_point(self):
        r = self.outcome((3, 14))
        self.assertEqual((r["outcome"], r["winner_id"]), ("out", 2))

    def test_net_gives_opponent_the_point(self):
        self.assertEqual(self.outcome((3, 6.7))["winner_id"], 2)

    def test_no_landing_awards_nothing(self):
        r = self.outcome(None)
        self.assertEqual((r["outcome"], r["winner_id"]), ("unknown", 0))
        self.assertEqual(r["score_after"], {"1": 0, "2": 0})
        self.assertFalse(r["score_reliable"])

    def test_clip_ending_mid_rally_is_unknown(self):
        self.assertEqual(self.outcome((3, 10), last=190, length=195)["outcome"], "unknown")

    def test_moving_shuttle_is_not_a_landing(self):
        rows, hits, shots = clip(None)
        for i in range(120, 150):
            rows[i] = frame(i, [3, 8 + (i - 120) * 0.2])
        self.assertEqual(build_rallies(rows, hits, shots, FPS)[0][0]["outcome"], "unknown")

    def test_long_gap_splits_rallies_and_score_carries(self):
        rows = [frame(i) for i in range(1000)]
        for start in (100, 400):
            for i in range(start + 30, start + 40):
                rows[i] = frame(i, [3, 10])
        hits = [hit(1, 50, 2), hit(2, 100, 1), hit(3, 350, 2), hit(4, 400, 1)]
        shots = [shot(50, "Bottom"), shot(100, "Top"), shot(350, "Bottom"), shot(400, "Top")]
        rallies, info = build_rallies(rows, hits, shots, FPS)
        self.assertEqual([r["rally_index"] for r in rallies], [1, 2])
        self.assertEqual(rallies[1]["score_before"], {"1": 1, "2": 0})
        self.assertEqual(rallies[1]["score_after"], {"1": 2, "2": 0})
        self.assertEqual(info["decided"], 2)

    def test_lone_hit_is_not_a_rally(self):
        rows = [frame(i) for i in range(400)]
        rallies, info = build_rallies(rows, [hit(1, 100, 1)], [], FPS)
        self.assertEqual((rallies, info["dropped_short_segments"]), ([], 1))

    def test_game_closes_at_21_and_score_resets(self):
        length, hits, shots, rows = 22 * 300, [], [], None
        rows = [frame(i) for i in range(length)]
        for n in range(22):
            base = 100 + n * 300
            hits += [hit(n * 2 + 1, base, 2), hit(n * 2 + 2, base + 50, 1)]
            shots += [shot(base, "Bottom"), shot(base + 50, "Top")]
            for i in range(base + 70, base + 80):
                rows[i] = frame(i, [3, 10])
        rallies, _ = build_rallies(rows, hits, shots, FPS)
        self.assertEqual(rallies[20]["score_after"], {"1": 21, "2": 0})
        self.assertEqual((rallies[21]["game"], rallies[21]["score_before"]), (2, {"1": 0, "2": 0}))

    def test_unknown_hitter_is_unknown(self):
        rows, hits, shots = clip((3, 10))
        hits[-1]["hitter_id"] = None
        self.assertEqual(build_rallies(rows, hits, shots, FPS)[0][0]["winner_id"], 0)


if __name__ == "__main__":
    unittest.main()
