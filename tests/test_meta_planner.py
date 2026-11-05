import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from optiloop.meta_planner import main, plan
from tests.test_agent import C


class PlannerTests(unittest.TestCase):
    def test_advisory_and_actual_submission(self):
        measured = dict(C, batch=4)
        history = [dict(requested_config=measured, docs_per_sec=1.2, split='tuning')]
        response = dict(content=json.dumps(dict(config=C, reason='Test batch hypothesis')),
                        http_status=200, model='muse-spark-1.3')
        with patch('optiloop.meta_planner.meta_response', return_value=response) as call:
            result = plan(history, [C, measured])
        sent = json.loads(call.call_args.args[0][1]['content'])
        self.assertEqual(sent['history'], history)
        self.assertEqual(sent['remaining'], [C])
        self.assertEqual(result['source'], 'meta_api')
        for field in ('benchmarked', 'quality_approved', 'deployed'):
            self.assertFalse(result[field])

    def test_invalid_proposal_never_falls_back(self):
        with patch('optiloop.meta_planner.meta_response', return_value=dict(content='{}')):
            with self.assertRaises(ValueError):
                plan([dict(docs_per_sec=1.2)], [C])

    def test_heldout_rejected_before_network(self):
        with patch('optiloop.meta_planner.meta_response') as call:
            with self.assertRaises(ValueError):
                plan([dict(split='heldout')], [C])
        call.assert_not_called()

    def test_missing_key_is_unverified(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            output = Path(tmp) / 'ping.json'
            with patch('builtins.print'):
                self.assertEqual(main(['--ping', '--output', str(output)]), 1)
            self.assertEqual(json.loads(output.read_text())['status'], 'unverified')

    def test_error_text_never_exposed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MODEL_API_KEY': 'private-test-key'}):
            output = Path(tmp) / 'ping.json'
            with patch('optiloop.meta_planner.meta_response', side_effect=RuntimeError('private-test-key')), patch('builtins.print'):
                self.assertEqual(main(['--ping', '--output', str(output)]), 1)
            self.assertNotIn('private-test-key', output.read_text())
