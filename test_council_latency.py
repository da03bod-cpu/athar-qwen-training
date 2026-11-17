import io
import json
import os
import sys
import threading
import unittest
from contextlib import redirect_stdout
from types import ModuleType
from unittest.mock import Mock, patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

import handler_advisory_council as council

with patch.dict(sys.modules, {"runpod": ModuleType("runpod")}):
    import handler


OPINION = (
    "تبدأ المراجعة بمطابقة الهدف مع نتائج البرامج الموثقة في الحالة. "
    "تحقق من تعريف المؤشر ومصدر القياس قبل اقتراح أي تغيير في طريقة جمع البيانات. "
    "إذا تعذر إثبات حالة التنفيذ فأحلها إلى المسؤول للتحقق من السجل الأصلي. "
    "يقتصر الرأي على مدى ملاءمة المؤشر وحدود الأدلة المقدمة."
)
REQUEST = {"type": "advisory_consultation", "topic": "interventions"}


class TestSpecialistScheduling(unittest.TestCase):
    def setUp(self):
        self.engine = council.AtharCouncilEngine.__new__(council.AtharCouncilEngine)
        self.selected = [{"advisor_id": f"ADVISOR-{i}", "advisor_name_ar": "مستشار"} for i in range(6)]
        self.engine._prepare_advisor = Mock(side_effect=lambda a, r, s: (
            a["advisor_id"], {"instruction": "تقييم الحالة", "case": "synthetic"},
        ))
        self.engine._generate_specialist_batch = Mock(
            side_effect=lambda prompts: [f"{identifier}: {OPINION}" for identifier, _ in prompts]
        )
        self.engine._generate = Mock(return_value=OPINION)

    def schedule(self, size):
        serial, batches = {}, []
        with patch.dict(os.environ, {"COUNCIL_SPECIALIST_BATCH_SIZE": str(size)}), redirect_stdout(io.StringIO()):
            outputs = self.engine._run_specialists(REQUEST, self.selected, serial, batches)
        return outputs, serial, batches

    def test_six_independent_advisors_reach_synthesis_in_order_with_three_calls(self):
        expected = [a["advisor_id"] for a in self.selected]
        self.engine._resolve_selected_advisors = Mock(return_value=self.selected)
        self.engine._resolve_meta_advisor = Mock(return_value={"slug": "META"})
        self.engine._run_meta = Mock(return_value={"synthetic_result": True})
        self.engine._validate_screen3_public_response = Mock()
        with patch.dict(os.environ, {"COUNCIL_SPECIALIST_BATCH_SIZE": "2"}), redirect_stdout(io.StringIO()):
            result = self.engine.consult(REQUEST)
        self.assertEqual(result, {"synthetic_result": True})
        opinions = self.engine._run_meta.call_args.args[1]
        self.assertEqual([a["advisor_id"] for a in opinions], expected)
        self.assertTrue(all(a["opinion"].startswith(a["advisor_id"] + ":") for a in opinions))
        self.assertEqual(self.engine._generate_specialist_batch.call_count, 3)
        self.engine._generate.assert_not_called()
        self.engine._validate_screen3_public_response.assert_called_once_with(result)

    def test_default_uses_legacy_single_advisor_path(self):
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            serial, batches = {}, []
            outputs = self.engine._run_specialists(REQUEST, self.selected, serial, batches)
        self.assertEqual(len(outputs), len(self.selected))
        self.assertEqual(set(serial), {a["advisor_id"] for a in self.selected})
        self.assertEqual(batches, [])
        self.assertEqual(self.engine._generate.call_count, 6)
        self.engine._generate_specialist_batch.assert_not_called()

    def test_only_the_poor_opinion_is_retried(self):
        self.selected = self.selected[:2]
        self.engine._generate_specialist_batch.side_effect = None
        self.engine._generate_specialist_batch.return_value = [OPINION, ""]
        outputs, _, _ = self.schedule(2)
        self.assertEqual(len(outputs), 2)
        self.engine._generate.assert_called_once()
        self.assertEqual(self.engine._generate.call_args.args[1], "ADVISOR-1")

    def test_large_batch_splits_and_preserves_outputs_after_cuda_oom(self):
        self.selected = self.selected[:4]

        def generate(prompts):
            if len(prompts) > 2:
                raise torch.cuda.OutOfMemoryError("simulated oversized batch")
            return [f"{identifier}: {OPINION}" for identifier, _ in prompts]

        self.engine._generate_specialist_batch.side_effect = generate
        outputs, _, batches = self.schedule(4)
        self.assertEqual([x["advisor_id"] for x in outputs], [x["advisor_id"] for x in self.selected])
        self.assertEqual(batches[0]["fallback"], "gpu_out_of_memory")
        self.assertEqual([x["size"] for x in batches[1:]], [2, 2])
        self.engine._generate.assert_not_called()

    def test_unrelated_generation_error_is_not_swallowed_or_retried(self):
        self.engine._generate_specialist_batch.side_effect = RuntimeError("adapter mismatch")
        with self.assertRaisesRegex(RuntimeError, "adapter mismatch"):
            self.schedule(2)
        self.engine._generate.assert_not_called()


class SyntheticTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, return_tensors=None, **kwargs):
        ids = [int(t) for t in prompt.split()]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in ids if int(token) != self.eos_token_id)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in rows]


class TestTinyModel(unittest.TestCase):
    def test_greedy_continuations_match_serial_with_left_padding_and_lora(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(37)
            model = Qwen3ForCausalLM(Qwen3Config(
                vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                bos_token_id=1, eos_token_id=0, pad_token_id=0,
            ))
            model = get_peft_model(model, LoraConfig(
                r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM", init_lora_weights=False,
            ), adapter_name="specialist").eval()
        engine = council.AtharCouncilEngine.__new__(council.AtharCouncilEngine)
        engine._render_chat = lambda system, user: user
        engine.tokenizer = SyntheticTokenizer()
        engine.model = model
        engine.model_lock = threading.RLock()
        prompts = [("", "1 7 0 8"), ("", "1 12 34 51 21 9 4")]
        with patch.object(council, "ADVISOR_MAX_NEW_TOKENS", 10):
            batch = engine._generate_specialist_batch(prompts)
            serial = [engine._generate(
                "specialist", system, user, 10, deterministic=True,
                repetition_penalty=1.10, no_repeat_ngram_size=8,
            ) for system, user in prompts]
        self.assertTrue(all(serial))
        self.assertEqual(batch, serial)

        with patch.object(council, "SPECIALIST_BATCH_MAX_PADDED_TOKENS", 13):
            with self.assertRaises(council._SpecialistBatchTooWide):
                engine._generate_specialist_batch(prompts)
        with patch.object(council, "MAX_MODEL_INPUT_TOKENS", 6):
            with self.assertRaisesRegex(ValueError, "NOT truncated"):
                engine._generate_specialist_batch(prompts)


class TestWorkerTrace(unittest.TestCase):
    def test_failed_application_result_is_correlated_without_request_body(self):
        result = {"status": "FAILED", "error": "synthetic failure"}
        request = {**REQUEST, "athar_trace_id": "73526e22-a84f-480d-8ef6-d0faebdf2883", "payload": {"secret_marker": "private text"}}
        output = io.StringIO()
        with patch.object(handler, "_handler_impl", return_value=result), redirect_stdout(output):
            self.assertIs(handler.handler({"id": "sync-xyz-e2", "input": request}), result)
        log = output.getvalue()
        self.assertNotIn("secret_marker", log)
        self.assertNotIn("private text", log)
        start, finish = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines()]
        self.assertEqual(start["event"], "worker_started")
        self.assertEqual(finish["event"], "worker_finished")
        self.assertEqual(finish["outcome"], "application_failed")
        self.assertEqual(finish["runpod_job_id"], "sync-xyz-e2")
        self.assertEqual(finish["athar_trace_id"], request["athar_trace_id"])

    def test_non_council_and_untrusted_trace_fields_are_excluded(self):
        self.assertIsNone(handler._council_trace({"input": {"type": "knowledge_index"}}))
        self.assertIsNone(handler._council_trace({"input": {**REQUEST, "preflight": True}}))
        trace = handler._council_trace({"id": "unsafe\nbody", "input": {**REQUEST, "athar_trace_id": "private text"}})
        self.assertIsNone(trace["runpod_job_id"])
        self.assertNotIn("athar_trace_id", trace)


if __name__ == "__main__":
    unittest.main()
