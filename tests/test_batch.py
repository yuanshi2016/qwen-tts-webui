"""Small checks for adaptive batch handling."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from qwen_tts_webui.api.api import iter_batches, normalize_texts
from qwen_tts_webui.backend.memory_manager import estimate_batch_capacity


class BatchTest(TestCase):
    def test_text_normalization_and_chunking(self) -> None:
        texts = normalize_texts([" a ", "", "b", "c"], False)
        self.assertEqual(texts, ["a", "b", "c"])
        self.assertEqual(list(iter_batches(texts, 2)), [["a", "b"], ["c"]])

    @patch("qwen_tts_webui.backend.memory_manager.psutil.virtual_memory")
    @patch("qwen_tts_webui.backend.memory_manager.get_free_memory")
    @patch("qwen_tts_webui.backend.memory_manager.get_torch_device")
    def test_loaded_model_allows_larger_batch(self, get_device, get_free, virtual_memory) -> None:
        get_device.return_value = torch.device("cpu")
        get_free.return_value = 16 * 1024**3
        virtual_memory.return_value = SimpleNamespace(total=32 * 1024**3)

        unloaded = estimate_batch_capacity("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", torch.bfloat16)
        loaded = estimate_batch_capacity("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", torch.bfloat16, model_loaded=True)

        self.assertEqual(unloaded["model_size_billion"], 1.7)
        self.assertGreater(loaded["recommended_batch_size"], unloaded["recommended_batch_size"])