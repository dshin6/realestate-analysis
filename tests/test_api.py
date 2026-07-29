import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from realestate_analysis.api import PublicDataApiError, iter_months, load_trades, parse_trade_xml
from realestate_analysis.config import TARGET_COMPLEX


SUCCESS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<response><header><resultCode>000</resultCode><resultMsg>OK</resultMsg></header>
<body><items><item>
<aptNm>동탄시범한빛마을한화꿈에그린</aptNm><dealAmount> 65,000 </dealAmount>
<dealYear>2026</dealYear><dealMonth>7</dealMonth><dealDay>3</dealDay>
<excluUseAr>84.8</excluUseAr><floor>12</floor><aptDong>231동</aptDong>
<umdNm>반송동</umdNm><jibun>21</jibun><cdealDay></cdealDay>
</item></items><numOfRows>10</numOfRows><pageNo>1</pageNo><totalCount>1</totalCount></body></response>"""


class ApiParsingTest(unittest.TestCase):
    def test_month_range_includes_both_ends(self):
        self.assertEqual(list(iter_months("202512", "202602")), ["202512", "202601", "202602"])

    def test_parses_trade_and_converts_manwon_to_won(self):
        rows, total = parse_trade_xml(SUCCESS_XML)
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["price_won"], 650_000_000)
        self.assertEqual(rows[0]["deal_date"], "2026-07-03")
        self.assertEqual(rows[0]["building"], "231동")
        self.assertFalse(rows[0]["cancelled"])

    def test_raises_for_api_error(self):
        xml = "<response><header><resultCode>30</resultCode><resultMsg>KEY ERROR</resultMsg></header></response>"
        with self.assertRaises(PublicDataApiError):
            parse_trade_xml(xml)

    def test_load_trades_uses_current_seed_without_api_calls(self):
        rows = [
            {
                "deal_date": "2026-07-03",
                "price_won": 650_000_000,
                "exclusive_area": 84.8,
                "floor": 12,
                "apartment_name": TARGET_COMPLEX.name,
                "building": "231동",
                "legal_dong": "반송동",
                "jibun": "21",
                "cancelled": False,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_path = root / "data" / "seed" / "trades.json"
            seed_path.parent.mkdir(parents=True)
            seed_path.write_text(
                json.dumps({"fetched_at": "seed", "end_ym": "202607", "rows": rows}),
                encoding="utf-8",
            )
            with patch("realestate_analysis.api.fetch_month") as fetch_month:
                frame, fetched_at = load_trades(
                    service_key="key",
                    config=TARGET_COMPLEX,
                    end_ym="202607",
                    cache_path=root / "data" / "cache" / "trades.json",
                    seed_path=seed_path,
                )

        fetch_month.assert_not_called()
        self.assertEqual(len(frame), 1)
        self.assertEqual(fetched_at, "seed")

    def test_load_trades_fetches_only_months_after_seed(self):
        old_row = {
            "deal_date": "2026-06-10",
            "price_won": 600_000_000,
            "exclusive_area": 84.8,
            "floor": 10,
            "apartment_name": TARGET_COMPLEX.name,
            "building": "231동",
            "legal_dong": "반송동",
            "jibun": "21",
            "cancelled": False,
        }
        new_row = {
            **old_row,
            "deal_date": "2026-07-10",
            "price_won": 660_000_000,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_path = root / "data" / "seed" / "trades.json"
            seed_path.parent.mkdir(parents=True)
            seed_path.write_text(
                json.dumps({"fetched_at": "seed", "end_ym": "202606", "rows": [old_row]}),
                encoding="utf-8",
            )
            with patch("realestate_analysis.api.fetch_month", return_value=[new_row]) as fetch_month:
                frame, _ = load_trades(
                    service_key="key",
                    config=TARGET_COMPLEX,
                    end_ym="202607",
                    cache_path=root / "data" / "cache" / "trades.json",
                    seed_path=seed_path,
                )

        self.assertEqual(fetch_month.call_count, 1)
        self.assertEqual(fetch_month.call_args.args[2], "202607")
        self.assertEqual(len(frame), 2)


if __name__ == "__main__":
    unittest.main()
