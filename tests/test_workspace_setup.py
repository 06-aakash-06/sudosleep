"""Verify clean-clone diagnostics and direct-training artifact compatibility."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_workspace import check_layout


class WorkspaceSetupTest(unittest.TestCase):
    def test_missing_raw_data_returns_actionable_failed_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            checks = check_layout(Path(directory), 'raw')
        self.assertFalse(all(ok for ok, _ in checks))
        self.assertTrue(any('0/100' in message for _, message in checks))
        self.assertTrue(any('participant_info' in message for _, message in checks))

    def test_overview_works_without_optional_supervisor_status(self):
        import research_api
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dreamt_results.json').write_text(json.dumps({'test': {'n_windows': 2}}))
            (root / 'subject_splits.json').write_text(json.dumps({'test': ['S001']}))
            (root / 'audit_64hz.csv').write_text('rows,elapsed_h,gaps,invalid_acc_rows\n100,1,0,0\n')
            (root / 'test_predictions.csv').write_text('subject,label,score\nS001,0,0.1\nS001,1,0.9\n')
            with patch.object(research_api, 'DERIVED', root):
                result = research_api.overview()
            self.assertEqual(result['status']['stage'], 'complete')
            self.assertEqual(result['audit']['samples'], 100)
            self.assertTrue(result['curves']['roc'])
            self.assertFalse((root / 'overnight_status.json').exists())


if __name__ == '__main__':
    unittest.main()
