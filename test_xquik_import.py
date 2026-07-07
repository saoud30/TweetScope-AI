from io import BytesIO
import unittest

from xquik_import import XquikImportError, load_xquik_texts


class XquikImportTest(unittest.TestCase):
    def test_loads_csv_text_column(self):
        payload = BytesIO(b"text,id\nFirst tweet,1\nSecond tweet,2\n")

        self.assertEqual(load_xquik_texts(payload), ["First tweet", "Second tweet"])

    def test_loads_nested_json_export(self):
        payload = BytesIO(
            b'{"data":[{"tweet":{"full_text":"Nested tweet"}},{"content":"Second tweet"}]}'
        )

        self.assertEqual(load_xquik_texts(payload), ["Nested tweet", "Second tweet"])

    def test_rejects_export_without_text(self):
        payload = BytesIO(b"id,url\n1,https://example.com\n")

        with self.assertRaises(XquikImportError):
            load_xquik_texts(payload)


if __name__ == "__main__":
    unittest.main()
