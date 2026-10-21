import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_exp003_when_idle import available_gpu, sha256, validate_result


class Exp003QueueTest(unittest.TestCase):
    def test_waits_for_existing_compute_jobs_even_when_memory_available(self):
        with patch('run_exp003_when_idle.subprocess.check_output', side_effect=[
            '0, GPU-a, 24000, 0\n1, GPU-b, 24000, 0\n', 'GPU-a\nGPU-b\n',
        ]):
            self.assertIsNone(available_gpu())

    def test_chooses_only_idle_gpu_with_sufficient_memory(self):
        with patch('run_exp003_when_idle.subprocess.check_output', side_effect=[
            '0, GPU-a, 2900, 0\n1, GPU-b, 24000, 0\n', 'GPU-a\n',
        ]):
            self.assertEqual(available_gpu(), '1')

    def test_smoke_cannot_unlock_formal_without_all_six_reference_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'config.json'
            config.write_text('{}')
            records = root / 'stage_a_leave_one_out_per_episode.json'
            records.write_text(json.dumps({'records': [{}]}))
            manifest = {'git': {'commit': 'signed', 'clean': True, 'branch': None},
                        'inputs': {k: {'sha256': k} for k in ('episodes', 'checkpoint', 'sam2_base')}}
            (root / 'run_manifest.json').write_text(json.dumps(manifest))
            metrics = {'git_sha': 'signed', 'evidence_label': 'DIAGNOSTIC_ONLY',
                       'episode_count': 1, 'unique_episode_ids': 1, 'loaded_adapter_tensors': 240,
                       'execution_config_sha256': sha256(config), 'episode_manifest_sha256': 'episodes',
                       'adapter_sha256': 'checkpoint', 'sam2_checkpoint_sha256': 'sam2_base',
                       'per_episode_sha256': sha256(records), 'cached_reference_checks': []}
            path = root / 'stage_a_leave_one_out_metrics.json'
            path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(RuntimeError, 'Missing six-way'):
                validate_result(root, config, 'signed', True)
            check = {'passed': True, 'rows': {k: {'binary_mask_exact': True}
                     for k in ('B0', 'D0', 'D1', 'D2', 'D3', 'D4')}}
            metrics['cached_reference_checks'] = [check]
            path.write_text(json.dumps(metrics))
            validate_result(root, config, 'signed', True)
            check['rows']['D4']['binary_mask_exact'] = False
            path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(RuntimeError, 'prediction mismatch'):
                validate_result(root, config, 'signed', True)
