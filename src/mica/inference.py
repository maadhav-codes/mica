import gc
import resource
import time

from .config import ChatError, check_platform


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return min(n, max(0, len(b) - 1))


class Engine:
    def __init__(self):
        self.model = self.tokenizer = self.spec = self.cache = None
        self.cached_tokens = []
        self.stats = {}

    def invalidate(self):
        self.cache = None
        self.cached_tokens = []

    def close(self):
        self.invalidate()
        self.model = self.tokenizer = self.spec = None
        gc.collect()
        import mlx.core as mx
        mx.synchronize()
        mx.clear_cache()

    def load(self, spec, settings):
        check_platform()
        import mlx.core as mx
        from mlx_lm import load
        self.close()
        device = mx.device_info()
        # A guideline, not an allocation guarantee or system-wide setting.
        limit = min(int(settings.memory_gb * 1024**3), int(device["memory_size"] * 0.6))
        if spec.size >= limit * 0.8:
            raise ChatError("Model weights leave insufficient estimated headroom for KV cache and generation. Select a smaller native MLX 4-bit model.")
        mx.set_memory_limit(limit)
        mx.set_cache_limit(256 * 1024**2)
        mx.reset_peak_memory()
        start = time.perf_counter()
        try:
            self.model, self.tokenizer = load(spec.path, tokenizer_config={"trust_remote_code": False, "local_files_only": True})
            if not self.tokenizer.chat_template:
                raise ChatError("Tokenizer has no official chat template.")
            # Validate system/user role formatting before offering the prompt.
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": settings.system}, {"role": "user", "content": "Hello"}] if settings.system else [{"role": "user", "content": "Hello"}],
                tokenize=True, add_generation_prompt=True,
            )
            mx.eval(self.model.parameters())
            self.spec = spec
            self.load_seconds = time.perf_counter() - start
            self.load_peak_mlx_bytes = mx.get_peak_memory()
        except BaseException:
            self.close()
            raise

    def stream(self, tokens, settings):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        prefix = 0
        if self.cache is not None and can_trim_prompt_cache(self.cache):
            prefix = common_prefix(self.cached_tokens, tokens)
            if prefix:
                trim_prompt_cache(self.cache, len(self.cached_tokens) - prefix)
        if not prefix:
            self.cache = make_prompt_cache(self.model)
        self.cached_tokens = []
        mx.random.seed(settings.seed)
        mx.reset_peak_memory()
        start = time.perf_counter()
        first = None
        response = None
        complete = False
        iterator = stream_generate(
            self.model, self.tokenizer, tokens[prefix:], max_tokens=settings.max_tokens,
            sampler=make_sampler(temp=settings.temperature, top_p=settings.top_p),
            prompt_cache=self.cache, prefill_step_size=256,
        )
        try:
            for response in iterator:
                if first is None:
                    first = time.perf_counter() - start
                yield response.text
            complete = True
        finally:
            iterator.close()
            elapsed = time.perf_counter() - start
            self.stats = {
                "ttft_seconds": first,
                "elapsed_seconds": elapsed,
                "formatted_prompt_tokens": len(tokens),
                "cached_prompt_tokens": prefix,
                "processed_prompt_tokens": len(tokens) - prefix,
                "prompt_tokens_per_second": response.prompt_tps if response else None,
                "generation_tokens": response.generation_tokens if response else 0,
                "generation_tokens_per_second": response.generation_tps if response else None,
                "peak_mlx_bytes": mx.get_peak_memory(),
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "finish_reason": response.finish_reason if response and complete else "cancelled",
            }
            # Retain only the verified prompt prefix. Never infer EOS or retokenize output.
            if complete and can_trim_prompt_cache(self.cache):
                offsets = [getattr(c, "offset", None) for c in self.cache]
                if offsets and all(o == offsets[0] for o in offsets) and offsets[0] >= len(tokens):
                    trim_prompt_cache(self.cache, offsets[0] - len(tokens))
                    self.cached_tokens = list(tokens)
                else:
                    self.invalidate()
            else:
                self.invalidate()
            mx.clear_cache()
