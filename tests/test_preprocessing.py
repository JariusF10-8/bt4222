"""Checks for the shared handoff: splitting, preservation and CSV round trips."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import warnings

from src.data_loading import load_partition
from src.preprocessing import build_dataset


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.output = self.root / 'processed/v1'
        for source, events in [
            ('MINDsmall_train', [
                ['1', 'U1', '11/12/2019 11:59:59 PM', 'N1', 'N2-0 N1-1 N99-0'],
                ['2', 'U1', '11/13/2019 12:00:00 AM', '', 'N1-0 N2-1'],
            ]),
            ('MINDsmall_dev', [
                ['1', 'U2', '11/15/2019 12:00:00 AM', 'N1', 'N2-1 N1-0'],
            ]),
        ]:
            folder = self.raw / source
            folder.mkdir(parents=True)
            with (folder / 'behaviors.tsv').open('w') as stream:
                csv.writer(stream, delimiter='\t', quoting=csv.QUOTE_NONE).writerows(events)
            annotations = json.dumps([{'WikidataId': 'Q1', 'Label': 'A "quote"'}])
            rows = [
                ['N1', 'news', 'local', '"Quoted", title', 'NA', 'https://example.org', annotations, '[]'],
                ['N2', '', '', 'Other title', '', '', '[]', ''],
            ]
            # Real source contains literal quotes, not CSV-escaped TSV fields.
            (folder / 'news.tsv').write_text(''.join('\t'.join(row) + '\n' for row in rows))

    def test_complete_handoff_round_trip(self):
        summary = build_dataset(self.raw, self.output)
        self.assertEqual(summary['events'].tolist(), [1, 1, 1])
        train, candidates, news = load_partition('train', self.output)
        validation, _, _ = load_partition('validation', self.output)
        test, _, _ = load_partition('test', self.output)
        self.assertEqual(train.iloc[0]['history_ids'], ['N1'])
        self.assertEqual(validation.iloc[0]['history_ids'], [])
        self.assertTrue(validation.iloc[0]['history_missing'])
        self.assertNotEqual(train.iloc[0]['event_key'], test.iloc[0]['event_key'])
        self.assertEqual(candidates['news_id'].tolist(), ['N2', 'N1', 'N99'])
        self.assertEqual(candidates['label'].tolist(), [0, 1, 0])
        self.assertEqual(candidates['candidate_position'].tolist(), [0, 1, 2])
        indexed = news.set_index('news_id')
        self.assertEqual(indexed.loc['N1', 'title'], '"Quoted", title')
        self.assertEqual(indexed.loc['N1', 'abstract'], 'NA')
        self.assertEqual(indexed.loc['N1', 'title_wikidata_ids'], ['Q1'])
        self.assertTrue(indexed.loc['N2', 'abstract_missing'])
        self.assertTrue(indexed.loc['N99', 'metadata_missing'])
        self.assertEqual(indexed.loc['N99', 'title_entities_parsed'], [])
        self.assertTrue(hasattr(train['timestamp'].dt, 'year'))
        self.assertTrue(json.loads((self.output / 'manifest.json').read_text())['complete'])
        with self.assertRaises(FileExistsError):
            build_dataset(self.raw, self.output)

    def test_missing_dev_requires_explicit_partial_build(self):
        import shutil
        shutil.rmtree(self.raw / 'MINDsmall_dev')
        with self.assertRaises(FileNotFoundError):
            build_dataset(self.raw, self.output)
        with warnings.catch_warnings(record=True) as caught:
            build_dataset(self.raw, self.output, require_test=False)
        self.assertTrue(caught)
        manifest = json.loads((self.output / 'manifest.json').read_text())
        self.assertFalse(manifest['complete'])
        self.assertEqual(manifest['partitions'], ['train', 'validation'])
        self.assertFalse((self.output / 'test').exists())
        with self.assertRaises(ValueError):
            load_partition('test', self.output)

    def test_malformed_metadata_is_not_published(self):
        path = self.raw / 'MINDsmall_train/news.tsv'
        path.write_text(path.read_text().replace('[]', '{broken', 1))
        with self.assertRaises(ValueError):
            build_dataset(self.raw, self.output)
        self.assertFalse(self.output.exists())

    def test_temporal_overlap_is_rejected(self):
        path = self.raw / 'MINDsmall_dev/behaviors.tsv'
        path.write_text(path.read_text().replace('11/15/2019', '11/12/2019'))
        with self.assertRaisesRegex(ValueError, 'Times overlap'):
            build_dataset(self.raw, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
