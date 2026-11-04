"""Pipeline-level tests: disk cache of model results and the full JSON contract."""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.data as D  # noqa: E402
from engine import pipeline as P  # noqa: E402
from tests.test_engine import synthetic  # noqa: E402


class PipelineCache(unittest.TestCase):
    def setUp(self):
        self.df = synthetic(1500, seed=9)
        self.saved = (D.load, D.CACHE_DIR, D.DEMO, P.headlines, P.macro_peek, P.crypto_regime_peek)
        D.CACHE_DIR = tempfile.mkdtemp()
        D.DEMO = False

        def fake_load(inst, bars=5000, max_age=240):
            if inst.symbol == "ZZZ":
                return self.df, "test"
            raise D.DataError("no benchmark in tests")
        D.load = fake_load
        P.headlines = lambda *a, **k: {"items": [], "n": 0, "score": 0.0}
        P.macro_peek = lambda: None
        P.crypto_regime_peek = lambda: None

    def tearDown(self):
        D.load, D.CACHE_DIR, D.DEMO, P.headlines, P.macro_peek, P.crypto_regime_peek = self.saved

    def test_second_run_hits_disk_cache_with_identical_results(self):
        t = time.time()
        a = P._analyze("ZZZ", "crypto", "1d", 1, 5000)
        first = time.time() - t
        t = time.time()
        b = P._analyze("ZZZ", "crypto", "1d", 1, 5000)
        second = time.time() - t
        self.assertEqual(a["signal"]["p_up"], b["signal"]["p_up"])
        self.assertEqual(a["validation"]["accuracy"], b["validation"]["accuracy"])
        self.assertLess(second, first / 2, (first, second))

    def test_output_is_strict_json(self):
        import server
        out = server._clean(P._analyze("ZZZ", "crypto", "1d", 1, 5000))
        json.dumps(out, allow_nan=False)
        for k in ("signal", "analysis", "backtest", "validation", "ohlcv"):
            self.assertIn(k, out)
        self.assertEqual(sum(s["p"] for s in out["analysis"]["scenarios"]), 100)


if __name__ == "__main__":
    unittest.main()
