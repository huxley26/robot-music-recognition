"""不加载模型或硬件SDK，独立执行数据划分与进程接口函数。"""
import ast
import collections
import hashlib
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import Mock
import os

ROOT = Path(__file__).resolve().parents[1]


def function_namespace(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text('utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), relative, 'exec'), namespace)
    return namespace


class DataSplitTests(unittest.TestCase):
    def test_sources_stay_disjoint_and_noise_is_train_only(self):
        ns = function_namespace('training/train.py',
                                {'stratified_split_by_src', 'training_noise_rows'},
                                {'defaultdict': collections.defaultdict, 'random': random})
        rows = [{'class': cls, 'src': f'{cls}/{source}.wav', 'out': f'{cls}/{source}-{clip}.wav'}
                for cls in ['dream', 'noises'] for source in range(10) for clip in range(3)]
        train, val, test = ns['stratified_split_by_src'](rows)
        groups = [{row['src'] for row in part} for part in (train, val, test)]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(len(train) + len(val) + len(test), len(rows))
        noise = ns['training_noise_rows'](train)
        self.assertTrue(noise)
        self.assertTrue(all(row['class'] == 'noises' for row in noise))
        self.assertFalse({row['src'] for row in noise} & (groups[1] | groups[2]))


class RobotInterfaceTests(unittest.TestCase):
    def namespace(self):
        process = Mock()
        process.PIPE = -1
        ns = {'os': os, 'sys': sys, 'subprocess': process,
              'song_scripts': {'dream': 'Huawei_actions'}, 'current_song': None,
              'current_process': None}
        return function_namespace('robot/music_control.py', {'start_dance'}, ns)

    def test_selected_script_directory_is_used(self):
        ns = self.namespace()
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-") as folder:
            path = Path(folder) / 'Huawei_actions.py'
            path.write_text('# external choreography\n', 'utf-8')
            self.assertTrue(ns['start_dance']('dream', folder))
            args, kwargs = ns['subprocess'].Popen.call_args
            self.assertEqual(args[0], [sys.executable, str(path)])
            self.assertNotIn('stdout', kwargs)
            self.assertNotIn('stderr', kwargs)

    def test_missing_script_does_not_launch(self):
        ns = self.namespace()
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test-") as folder:
            self.assertFalse(ns['start_dance']('dream', folder))
        ns['subprocess'].Popen.assert_not_called()
        self.assertIsNone(ns['current_song'])



class ArtifactTests(unittest.TestCase):
    def test_model_checksum(self):
        digest = (ROOT / 'models/SHA256SUMS').read_text().split()[0]
        self.assertEqual(hashlib.sha256((ROOT / 'models/music_classifier.pt').read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()

