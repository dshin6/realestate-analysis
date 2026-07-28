import unittest

from realestate_analysis.api import PublicDataApiError, iter_months, parse_trade_xml


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


if __name__ == "__main__":
    unittest.main()

