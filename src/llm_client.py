"""Unified LLM interface with swappable backends.

Backends:
  - api           → hosted HTTP API (dev only; no internet on Kaggle scoring)
  - ollama        → local Ollama server for laptop smoke-tests
  - kaggle_local  → in-process HuggingFace model (4-bit) for Kaggle
  - mock          → deterministic stub for wiring tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LLMClient:
    def __init__(self, backend: str, model_name: str, **kwargs: Any):
        if backend not in ("api", "ollama", "kaggle_local", "mock"):
            raise ValueError(f"Unknown backend: {backend}")
        self.backend = backend
        self.model_name = model_name
        self.kwargs = kwargs
        self._hf_model = None
        self._hf_tokenizer = None
        self._mock_call = 0

        if backend == "kaggle_local" and not kwargs.get("lazy_load", False):
            self._init_kaggle_local()

    @classmethod
    def from_config(cls, config: dict | None = None, config_path: str | Path | None = None) -> "LLMClient":
        if config is None:
            config = load_config(config_path)
        backend = config["backend"]
        model_name = config["model_name"]
        extra = dict(config.get(backend, {}) or {})
        # Allow top-level overrides used by kaggle config files
        for key in ("load_in_4bit", "device_map", "max_memory", "torch_dtype", "lazy_load"):
            if key in config and key not in extra:
                extra[key] = config[key]
        return cls(backend=backend, model_name=model_name, **extra)

    def _init_kaggle_local(self) -> None:
        """Load HF CausalLM, optionally 4-bit via bitsandbytes."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "kaggle_local backend requires transformers, torch, bitsandbytes, accelerate"
            ) from e

        load_in_4bit = bool(self.kwargs.get("load_in_4bit", True))
        device_map = self.kwargs.get("device_map", "auto")
        max_memory = self.kwargs.get("max_memory")
        trust_remote_code = bool(self.kwargs.get("trust_remote_code", True))
        local_files_only = bool(self.kwargs.get("local_files_only", False))

        quant = None
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "load_in_4bit=True requires CUDA. Install a CUDA build of PyTorch "
                    "or set kaggle_local.load_in_4bit: false for CPU smoke (very slow)."
                )
            compute = self.kwargs.get("bnb_4bit_compute_dtype", "float16")
            compute_dtype = getattr(torch, compute) if isinstance(compute, str) else compute
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.kwargs.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(self.kwargs.get("bnb_4bit_use_double_quant", True)),
            )

        print(f"[kaggle_local] loading tokenizer: {self.model_name}")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        dtype = self.kwargs.get("torch_dtype")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "device_map": device_map,
        }
        if quant is not None:
            load_kwargs["quantization_config"] = quant
        elif dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if max_memory is not None:
            # YAML may use int GPU ids as keys; accelerate wants int | "cpu" | "disk"
            load_kwargs["max_memory"] = {
                (int(k) if str(k).isdigit() else k): v for k, v in dict(max_memory).items()
            }

        print(
            f"[kaggle_local] loading model 4bit={load_in_4bit} "
            f"device_map={device_map} cuda={torch.cuda.is_available()}"
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self._hf_model.eval()
        print("[kaggle_local] model ready")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        if self.backend == "ollama":
            return self._generate_ollama(prompt, max_tokens, temperature)
        if self.backend == "api":
            return self._generate_api(prompt, max_tokens, temperature)
        if self.backend == "kaggle_local":
            return self._generate_kaggle(prompt, max_tokens, temperature)
        if self.backend == "mock":
            return self._generate_mock(prompt, max_tokens, temperature)
        raise RuntimeError(f"Unhandled backend: {self.backend}")

    def _generate_mock(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Deterministic stub for pipeline wiring tests (no model required)."""
        self._mock_call += 1
        lower = prompt.lower()
        transform = self._mock_detect_transform(prompt) or getattr(self, "_mock_transform", "rotate_180")
        self._mock_transform = transform
        code = self._mock_code_for(transform)
        hyp = self._mock_hypotheses_for(transform)

        if "rank the top 2" in lower or (("first:" in lower) and ("candidate" in lower)):
            return "FIRST: 1\nSECOND: 2\nBRIEF: Simplest general transform.\n"

        if "fix the code" in lower or "failure report" in lower:
            return f"```python\n{code}\n```\n"

        if "implement this hypothesized rule" in lower:
            import re

            quoted = re.search(
                r'Implement this hypothesized rule as a Python function:\s*"([^"]+)"',
                prompt,
                re.I,
            )
            hyp_text = (quoted.group(1) if quoted else "").lower()
            if "transpose" in hyp_text:
                code = self._mock_code_for("transpose")
            elif "90" in hyp_text and "counter" in hyp_text:
                code = self._mock_code_for("rotate_270")
            elif "270" in hyp_text:
                code = self._mock_code_for("rotate_270")
            elif "90" in hyp_text and "180" not in hyp_text:
                code = self._mock_code_for("rotate_90")
            elif "left-right" in hyp_text or ("horizontal" in hyp_text and "vertical" not in hyp_text):
                code = self._mock_code_for("reflect_horizontal")
            elif "top-bottom" in hyp_text or ("vertical" in hyp_text and "horizontal" not in hyp_text):
                code = self._mock_code_for("reflect_vertical")
            elif "180" in hyp_text:
                code = self._mock_code_for("rotate_180")
            else:
                code = self._mock_code_for(transform)
            return f"```python\n{code}\n```\n"

        if "propose" in lower and "hypothesis" in lower:
            return hyp

        return hyp + f"\n```python\n{code}\n```\n"

    @staticmethod
    def _mock_code_for(transform: str) -> str:
        mapping = {
            "rotate_180": (
                "def solve(grid):\n"
                "    return [row[::-1] for row in grid[::-1]]"
            ),
            "rotate_90": (
                "def solve(grid):\n"
                "    h, w = len(grid), len(grid[0])\n"
                "    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]"
            ),
            "rotate_270": (
                "def solve(grid):\n"
                "    h, w = len(grid), len(grid[0])\n"
                "    return [[grid[r][w - 1 - c] for r in range(h)] for c in range(w)]"
            ),
            "reflect_horizontal": (
                "def solve(grid):\n"
                "    return [row[::-1] for row in grid]"
            ),
            "reflect_vertical": (
                "def solve(grid):\n"
                "    return grid[::-1]"
            ),
            "transpose": (
                "def solve(grid):\n"
                "    h, w = len(grid), len(grid[0])\n"
                "    return [[grid[r][c] for r in range(h)] for c in range(w)]"
            ),
        }
        return mapping.get(transform, mapping["rotate_180"])

    @staticmethod
    def _mock_hypotheses_for(transform: str) -> str:
        primary = {
            "rotate_180": "Rotate the grid 180 degrees.",
            "rotate_90": "Rotate the grid 90 degrees clockwise.",
            "rotate_270": "Rotate the grid 90 degrees counter-clockwise (270 clockwise).",
            "reflect_horizontal": "Reflect the grid horizontally (left-right mirror).",
            "reflect_vertical": "Reflect the grid vertically (top-bottom mirror).",
            "transpose": "Transpose the grid (swap rows and columns).",
        }.get(transform, "Rotate the grid 180 degrees.")
        return (
            f"HYPOTHESIS 1: {primary}\n"
            "HYPOTHESIS 2: Apply a simple geometric remapping of all cells.\n"
            "HYPOTHESIS 3: Recolor objects without changing positions.\n"
        )

    @staticmethod
    def _mock_parse_grids(prompt: str) -> list:
        import re

        grids = []
        parts = re.split(r"Raw grid:\s*", prompt)
        for part in parts[1:]:
            rows = []
            for line in part.splitlines():
                line = line.strip()
                if not line or not re.match(r"^[\d\s]+$", line):
                    break
                rows.append([int(x) for x in line.split()])
            if rows:
                grids.append(rows)
        return grids

    @classmethod
    def _mock_detect_transform(cls, prompt: str) -> str | None:
        grids = cls._mock_parse_grids(prompt)
        pairs = []
        for i in range(0, len(grids) - 1, 2):
            pairs.append((grids[i], grids[i + 1]))
        if not pairs:
            return None

        def rot180(g):
            return [row[::-1] for row in g[::-1]]

        def rot90(g):
            h, w = len(g), len(g[0])
            return [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]

        def rot270(g):
            return rot90(rot90(rot90(g)))

        def refl_h(g):
            return [row[::-1] for row in g]

        def refl_v(g):
            return g[::-1]

        def transpose(g):
            h, w = len(g), len(g[0])
            return [[g[r][c] for r in range(h)] for c in range(w)]

        candidates = [
            ("rotate_180", rot180),
            ("rotate_90", rot90),
            ("rotate_270", rot270),
            ("reflect_horizontal", refl_h),
            ("reflect_vertical", refl_v),
            ("transpose", transpose),
        ]
        for name, fn in candidates:
            if all(fn(inp) == out for inp, out in pairs):
                return name
        return None

    def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import json
        import urllib.request

        base = self.kwargs.get("base_url", "http://localhost:11434").rstrip("/")
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        req = urllib.request.Request(
            f"{base}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")

    def _generate_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import json
        import os
        import urllib.error
        import urllib.request

        key_env = self.kwargs.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(key_env, "") or self.kwargs.get("api_key", "")
        if not api_key:
            raise RuntimeError(f"API key not found in env var {key_env}")

        provider = (self.kwargs.get("provider") or "").lower()
        base = self.kwargs.get("base_url", "https://api.openai.com/v1").rstrip("/")

        if provider == "gemini" or "generativelanguage.googleapis.com" in base:
            return self._generate_gemini(prompt, max_tokens, temperature, api_key)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{base}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"API HTTP {e.code}: {body}") from e
        return data["choices"][0]["message"]["content"]

    def _generate_gemini(
        self, prompt: str, max_tokens: int, temperature: float, api_key: str
    ) -> str:
        import json
        import urllib.error
        import urllib.parse
        import urllib.request

        model = self.model_name
        if model.startswith("models/"):
            model = model[len("models/") :]
        q = urllib.parse.urlencode({"key": api_key})
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?{q}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"Gemini HTTP {e.code}: {body}") from e

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(texts)

    def _hf_input_device(self):
        assert self._hf_model is not None
        try:
            return self._hf_model.get_input_embeddings().weight.device
        except Exception:
            return next(self._hf_model.parameters()).device

    def _format_kaggle_prompt(self, prompt: str) -> str:
        """Wrap user prompt with chat template when available (Qwen3-friendly)."""
        tok = self._hf_tokenizer
        assert tok is not None
        if not getattr(tok, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizers without enable_thinking
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _generate_kaggle(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import torch

        if self._hf_model is None or self._hf_tokenizer is None:
            self._init_kaggle_local()
        assert self._hf_model is not None and self._hf_tokenizer is not None

        text = self._format_kaggle_prompt(prompt)
        inputs = self._hf_tokenizer(text, return_tensors="pt")
        device = self._hf_input_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._hf_tokenizer.pad_token_id,
            "eos_token_id": self._hf_tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self._hf_model.generate(**inputs, **gen_kwargs)

        gen = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self._hf_tokenizer.decode(gen, skip_special_tokens=True)
