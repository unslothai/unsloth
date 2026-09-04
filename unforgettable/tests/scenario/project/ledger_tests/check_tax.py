import unittest

from ledger.tax import tax_on


class TaxTests(unittest.TestCase):
    def test_tax_on_hundred_dollars(self):
        # $100.00 → 10000 cents; 8.25% = 825 cents.
        self.assertEqual(tax_on(10000), 825)
