import unittest
from app import clean_text

class TestSentimentFunctions(unittest.TestCase):

    def test_clean_text(self):
        text = "Hello!!!"
        result = clean_text(text)
        self.assertEqual(result, "hello")

    def test_empty_text(self):
        text = ""
        result = clean_text(text)
        self.assertEqual(result, "")

    def test_numbers(self):
        text = "Movie 123"
        result = clean_text(text)
        self.assertIn("123", result)

if __name__ == '__main__':
    unittest.main()