"""Server robustness and macro caching behaviour."""

import os
import sys
import threading
import time
import unittest
from http import HTTPStatus

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402
import engine.macro as M  # noqa: E402


class ServerLogging(unittest.TestCase):
    def test_error_log_with_http_status_does_not_crash(self):
        h = server.Handler.__new__(server.Handler)
        h.log_message("code %d, message %s", HTTPStatus.NOT_FOUND, "File not found")
        h.log_error("code %d, message %s", HTTPStatus.NOT_FOUND, "File not found")


class NetworkErrors(unittest.TestCase):
    def test_dns_failure_becomes_short_offline_message(self):
        import requests
        from engine import data as D
        e = requests.ConnectionError("Failed to resolve 'api.exchange.coinbase.com' ([Errno 8] nodename nor servname provided, or not known)")
        msg = str(D.net_error("https://api.exchange.coinbase.com/products/BTC-USD/stats", e))
        self.assertIn("No internet connection", msg)
        self.assertLess(len(msg), 140)

    def test_get_wraps_request_exceptions(self):
        import requests
        from engine import data as D
        orig = requests.get
        def boom(*a, **k):
            raise requests.ConnectionError("nodename nor servname provided, or not known")
        requests.get = boom
        try:
            with self.assertRaises(D.DataError):
                D._get("https://api.exchange.coinbase.com/x")
        finally:
            requests.get = orig


class MacroCache(unittest.TestCase):
    def setUp(self):
        M._memo.clear()
        self.calls = []

    def slow(self):
        self.calls.append(1)
        time.sleep(0.6)
        return {"ok": True}

    def test_peek_never_blocks(self):
        build = lambda: M._memoize("peekkey", 3600, self.slow)
        t = time.time()
        self.assertIsNone(M.peek("peekkey", build))
        self.assertLess(time.time() - t, 0.1)
        time.sleep(0.8)
        self.assertEqual(M.peek("peekkey", build), {"ok": True})

    def test_concurrent_callers_share_one_build(self):
        out = []
        th = [threading.Thread(target=lambda: out.append(M._memoize("shared", 3600, self.slow))) for _ in range(5)]
        [x.start() for x in th]
        [x.join() for x in th]
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(out), 5)


if __name__ == "__main__":
    unittest.main()
