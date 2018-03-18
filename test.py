import unittest
from backend_keras import to_long_lang
from backend_keras import process_sentence
from backend_keras import load_obj
from backend_keras import load_model
import os


class TestBackendKeras(unittest.TestCase):

    def test_model_loading(self):
        self.assertIsNotNone(load_model("/home/sovietspy2/PycharmProjects/backend_keras/model2.h5"))

    def test_load_obj(self):
        with self.assertRaises(FileNotFoundError):
            load_obj('asd')
        self.assertEqual(os.path.exists('./vocab_to_int.pkl'), True)
        self.assertEqual(os.path.exists('./int_to_languages.pkl'), True)
        self.assertIsNotNone(load_obj('int_to_languages'))
        self.assertIsNotNone(load_obj('vocab_to_int'))

    def test_process_sentence(self):
        self.assertEqual(process_sentence('Extreme , test with many ; stuff !!!'), 'Extreme test with many stuff'.lower())

    def test_to_long_lang(self):
        self.assertEqual(to_long_lang('en'), 'english')
        self.assertEqual(to_long_lang('hu'), 'hungarian')
        self.assertEqual(to_long_lang('fr'), 'french')
        self.assertEqual(to_long_lang('de'), 'german')
        self.assertEqual(to_long_lang('981274jhsdjkas'), 'err')


if __name__ == '__main__':
    unittest.main()