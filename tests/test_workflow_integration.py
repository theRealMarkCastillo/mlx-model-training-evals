"""Offline end-to-end check with a locally constructed miniature Qwen model."""
import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import yaml
from mlx.utils import tree_flatten
from mlx_lm.models.qwen2 import Model, ModelArgs
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

from src.benchmark import run_benchmark_suite
from src.dataset import DATA_DIR
from src.evaluate import run_comprehensive_evaluation
from src.runs import latest_path
from src.train import run_training


class WorkflowIntegrationTests(unittest.TestCase):
    def test_local_training_fusion_benchmark_and_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base'
            base.mkdir()
            vocabulary = {token: i for i, token in enumerate(['<unk>', '<bos>', '<|endoftext|>'] + sorted(ByteLevel.alphabet()))}
            backend = Tokenizer(BPE(vocabulary, [], unk_token='<unk>'))
            backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
            backend.decoder = ByteLevelDecoder()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', bos_token='<bos>', eos_token='<|endoftext|>')
            tokenizer.chat_template = "{% for message in messages %}{{ '<' + message['role'] + '>' + ' ' + message['content'] + ' ' }}{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
            tokenizer.save_pretrained(base)
            config = dict(model_type='qwen2', hidden_size=64, num_hidden_layers=1, intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=len(vocabulary), num_key_value_heads=2, eos_token_id=2)
            model = Model(ModelArgs.from_dict(config))
            nn.quantize(model, group_size=32, bits=4)
            config['quantization'] = {'group_size': 32, 'bits': 4}
            (base / 'config.json').write_text(json.dumps(config))
            mx.save_safetensors(str(base/'model.safetensors'), dict(tree_flatten(model.parameters())))
            training_config = yaml.safe_load(Path('config/base.yaml').read_text())
            training_config.update(model=str(base), data=str(DATA_DIR), adapter_path=str(root/'adapters'), iters=1, num_layers=1, batch_size=1, val_batches=1, max_seq_length=8192, steps_per_report=1)
            training_config['lora_parameters'] = {'rank': 2, 'scale': 2., 'dropout': 0.}
            path = root/'train.yaml'
            path.write_text(yaml.safe_dump(training_config))
            training = run_training(config_path=str(path), output_dir=str(root))
            report = run_benchmark_suite(
                str(base), training.adapter_path, output_dir=str(root),
                fuse=True, runs=1, warmup=0, max_tokens=3, quality_samples=1,
            )
            for variant in ('base', 'lora', 'fused'):
                stats = report[f'{variant}_stats']
                self.assertEqual(len(stats['runs']), 1)
                self.assertGreater(stats['avg_prompt_tokens'], 0)
                self.assertGreater(stats['avg_ttft_seconds'], 0)
            evaluation = json.loads(Path(report['manifest']['quality_evaluation']).read_text())
            for variant in ('base', 'lora', 'fused'):
                metrics = evaluation['datasets']['test'][variant]
                self.assertEqual(metrics['num_samples'], 1)
                self.assertGreater(metrics['loss_tokens'], 0)
                self.assertEqual(len(metrics['sample_results']), 1)
            fused = Path(latest_path(root/'fused_model'))
            self.assertTrue((fused/'config.json').is_file())
            self.assertTrue(list(fused.glob('*.safetensors')))
            self.assertTrue((root/'latest_benchmark.json').is_file())
            full = run_comprehensive_evaluation(str(base), training.adapter_path, num_eval_samples=1, output_dir=str(root),
                                                max_tokens=3, shots=2, challenge=True, quiet=True)
            self.assertEqual(full['variants'], ['base', 'fewshot', 'lora'])
            self.assertEqual(len(full['datasets']), 4)
            fewshot = full['datasets']['test']['fewshot']
            self.assertGreater(fewshot['sample_results'][0]['prompt_tokens'], full['datasets']['test']['base']['sample_results'][0]['prompt_tokens'])
