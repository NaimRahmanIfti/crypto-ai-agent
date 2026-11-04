"""Macro data sources: FRED fallback chain and the official-source parsers,
exercised against realistic fake responses (no network)."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.data as D  # noqa: E402
import engine.macro as M  # noqa: E402


class FakeResp:
    def __init__(self, text="", js=None, status=200):
        self.text, self._js, self.status_code = text, js, status

    def json(self):
        return self._js


class FredFallback(unittest.TestCase):
    def setUp(self):
        self.saved = (D.CACHE_DIR, M._fred_download, dict(M.ALT))
        D.CACHE_DIR = tempfile.mkdtemp()
        M._fred_failed.clear()
        M._fred_down_until[0] = 0.0

    def test_breaker_trips_for_all_series_after_a_timeout(self):
        calls = []

        def fred_dead(series):
            calls.append(series)
            raise D.DataError("fred.stlouisfed.org did not answer in time.")
        M._fred_download = fred_dead
        ok = lambda _: pd.Series(np.arange(50.0), index=pd.date_range("2024-01-01", periods=50))
        M.ALT["SER_A"] = ("Official", ok)
        M.ALT["SER_B"] = ("Official", ok)
        M.fred("SER_A", max_age=0)
        M.fred("SER_B", max_age=0)
        self.assertEqual(calls, ["SER_A"])  # SER_B never waited on FRED

    def tearDown(self):
        D.CACHE_DIR, M._fred_download = self.saved[0], self.saved[1]
        M.ALT.clear(); M.ALT.update(self.saved[2])
        M._fred_failed.clear()
        M._fred_down_until[0] = 0.0

    def test_falls_back_when_fred_times_out_and_skips_fred_next_time(self):
        calls = {"fred": 0, "alt": 0}

        def fred_dead(series):
            calls["fred"] += 1
            raise D.DataError("fred.stlouisfed.org did not answer in time.")

        def alt(series):
            calls["alt"] += 1
            return pd.Series(np.arange(100.0), index=pd.date_range("2024-01-01", periods=100))
        M._fred_download = fred_dead
        M.ALT["TESTSER"] = ("Official", alt)
        s = M.fred("TESTSER", max_age=0)
        self.assertEqual(len(s), 100)
        self.assertEqual(calls, {"fred": 1, "alt": 1})
        M.fred("TESTSER", max_age=0)          # FRED is remembered as failed: straight to fallback
        self.assertEqual(calls["fred"], 1)

    def test_proxy_label_survives_the_disk_cache(self):
        M._fred_download = lambda s: (_ for _ in ()).throw(D.DataError("down"))

        def proxy(_):
            s = pd.Series(np.linspace(28, 29, 80), index=pd.date_range("2024-01-01", periods=80))
            s.attrs["proxy"] = "UUP ETF"
            return s
        M.ALT["DTWEXBGS"] = ("Nasdaq", proxy)
        M.fred("DTWEXBGS", max_age=0)
        cached = M.fred("DTWEXBGS")           # second call served from cache
        self.assertEqual(cached.attrs.get("proxy"), "UUP ETF")

    def test_raises_clear_error_when_everything_fails(self):
        M._fred_download = lambda s: (_ for _ in ()).throw(D.DataError("down"))
        with self.assertRaises(D.DataError) as cm:
            M.fred("NO_ALT_SERIES", max_age=0)
        self.assertIn("unavailable", str(cm.exception))


class OfficialParsers(unittest.TestCase):
    def setUp(self):
        self.orig_get = M.requests.get
        M._treasury_cache.clear()

    def tearDown(self):
        M.requests.get = self.orig_get

    def test_treasury_curve(self):
        csv = "Date,1 Mo,2 Yr,10 Yr\n11/01/2026,4.1,3.60,4.20\n10/31/2026,4.1,3.62,4.25\n"
        M.requests.get = lambda *a, **k: FakeResp(text=csv)
        spread = M._treasury("spread")
        self.assertAlmostEqual(spread.iloc[-1], 0.60, places=6)
        self.assertTrue(spread.index.is_monotonic_increasing)

    def test_vix(self):
        csv = "DATE,OPEN,HIGH,LOW,CLOSE\n10/30/2026,15,16,14,15.5\n10/31/2026,15.5,17,15,16.2\n"
        M.requests.get = lambda *a, **k: FakeResp(text=csv)
        self.assertAlmostEqual(M._vix(None).iloc[-1], 16.2)

    def test_effr(self):
        js = {"refRates": [{"effectiveDate": "2026-10-31", "percentRate": 3.88}, {"effectiveDate": "2026-10-30", "percentRate": 3.89}]}
        M.requests.get = lambda *a, **k: FakeResp(js=js)
        s = M._effr(None)
        self.assertAlmostEqual(s.iloc[-1], 3.88)

    def test_bls_skips_annual_average(self):
        js = {"status": "REQUEST_SUCCEEDED", "Results": {"series": [{"data": [
            {"year": "2026", "period": "M09", "value": "4.3"}, {"year": "2026", "period": "M08", "value": "4.2"},
            {"year": "2025", "period": "M13", "value": "4.0"}]}]}}
        M.requests.get = lambda *a, **k: FakeResp(js=js)
        s = M._bls("LNS14000000")(None)
        self.assertEqual(len(s), 2)
        self.assertAlmostEqual(s.iloc[-1], 4.3)


if __name__ == "__main__":
    unittest.main()
