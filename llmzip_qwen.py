import torch
import numpy as np
import os
import sys

try:
    from AC import arithmeticcoding
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from AC import arithmeticcoding


class LLMzipQwen:
    def __init__(self, model, tokenizer, device='cuda', max_window=50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_window = max_window
        self.min_free_vram = 512 * 1024 * 1024  # 0.5 GB
        self.vocab_size = self._resolve_vocab_size()
        # Max position limit: evict cache before exceeding model's max_position_embeddings
        config = self.model.config
        if hasattr(config, 'text_config'):
            config = config.text_config
        self.max_pos_len = int(getattr(config, 'max_position_embeddings', 32768) * 0.95)

    def _resolve_vocab_size(self):
        """Get vocab_size from config, handling multimodal models (e.g. Gemma 3) where it's nested."""
        config = self.model.config
        if hasattr(config, 'vocab_size'):
            return config.vocab_size
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'vocab_size'):
            return config.text_config.vocab_size
        # Fallback: infer from the model's lm_head output dimension
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head.out_features
        raise ValueError("Cannot determine vocab_size from model config or architecture")

    def _get_free_vram(self):
        """Return free VRAM in bytes, or inf for non-CUDA devices."""
        if self.device == 'cuda' and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free
        return float('inf')

    def _vram_is_safe(self):
        """Return True if at least min_free_vram bytes are available."""
        return self._get_free_vram() >= self.min_free_vram

    def _evict_cache(self):
        """Clear CUDA cache to reclaim VRAM."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def _get_probs(self, input_ids, past_key_values=None):
        """Forward pass returning (probs_numpy, past_key_values).

        Uses KV cache when past_key_values is provided.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1).cpu().to(torch.float32).numpy()[0]

        # Sanitize
        if np.isnan(probs).any() or np.isinf(probs).any():
            probs = np.ones_like(probs) / len(probs)
        probs_sum = np.sum(probs)
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        return probs, outputs.past_key_values

    def _probs_to_cumul(self, probs):
        """Convert a probability vector to a cumulative frequency table for arithmetic coding."""
        vocab_size = len(probs)
        scale = 1000000.0
        freqs = probs * scale
        freqs = np.nan_to_num(freqs, nan=1.0, posinf=1.0, neginf=1.0)
        freqs = freqs.astype(np.uint64)
        freqs = np.maximum(freqs, 1)

        total_freq = np.sum(freqs)
        if total_freq >= (1 << 30):
            freqs = (freqs / 2).astype(np.uint64)
            freqs = np.maximum(freqs, 1)

        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(freqs)
        return cumul

    def encode(self, text, output_file):
        """Tokenize text, append EOS, and arithmetic-encode to output_file."""
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)[0]
        eos = torch.tensor([self.tokenizer.eos_token_id], device=self.device)
        tokens = torch.cat([input_ids, eos])
        num_tokens = len(tokens)
        vocab_size = self.vocab_size

        print(f"Total tokens to encode: {num_tokens}")

        with open(output_file, 'wb') as f:
            bitout = arithmeticcoding.BitOutputStream(f)
            enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

            # First token: uniform distribution (no context available)
            uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
            enc.write(uniform_cumul, tokens[0].item())

            past_kv = None
            cache_len = 0

            for i in range(1, num_tokens):
                # Evict KV cache if VRAM is low or cache exceeds max position length
                if past_kv is not None and (not self._vram_is_safe() or cache_len >= self.max_pos_len):
                    past_kv = None
                    cache_len = 0
                    self._evict_cache()

                if past_kv is None:
                    # Full forward on the last max_window tokens, rebuilding cache
                    start = max(0, i - self.max_window)
                    context = tokens[start:i].unsqueeze(0)
                    probs, past_kv = self._get_probs(context)
                    cache_len = i - start
                else:
                    # Incremental: feed only the latest token using cached KV
                    last_token = tokens[i - 1:i].unsqueeze(0)
                    probs, past_kv = self._get_probs(last_token, past_kv)
                    cache_len += 1

                cumul = self._probs_to_cumul(probs)
                enc.write(cumul, tokens[i].item())

                if i % 10 == 0:
                    print(f"Encoded {i}/{num_tokens} tokens (cache: {cache_len})", end='\r')
                    sys.stdout.flush()

            enc.finish()
            bitout.close()

        print(f"\nEncoding complete.")

        original_size = len(text)
        compressed_size = os.path.getsize(output_file)
        bpc = (compressed_size * 8) / original_size if original_size > 0 else 0
        print(f"Original size: {original_size} chars")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Bits per character: {bpc:.4f}")

    def decode(self, input_file):
        """Arithmetic-decode input_file back to text, stopping at EOS."""
        if not os.path.exists(input_file):
            print("File not found.")
            return ""

        file_in = open(input_file, 'rb')
        bitin = arithmeticcoding.BitInputStream(file_in)
        dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

        vocab_size = self.vocab_size

        # First token: uniform distribution
        uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
        first_token = dec.read(uniform_cumul, vocab_size)

        decoded_tokens = [first_token]
        past_kv = None
        cache_len = 0

        while True:
            # Evict KV cache if VRAM is low or cache exceeds max position length
            if past_kv is not None and (not self._vram_is_safe() or cache_len >= self.max_pos_len):
                past_kv = None
                cache_len = 0
                self._evict_cache()

            if past_kv is None:
                # Full forward on the last max_window tokens, rebuilding cache
                start = max(0, len(decoded_tokens) - self.max_window)
                context = torch.tensor([decoded_tokens[start:]], device=self.device)
                probs, past_kv = self._get_probs(context)
                cache_len = len(decoded_tokens) - start
            else:
                # Incremental: feed only the latest token using cached KV
                last_token = torch.tensor([[decoded_tokens[-1]]], device=self.device)
                probs, past_kv = self._get_probs(last_token, past_kv)
                cache_len += 1

            cumul = self._probs_to_cumul(probs)

            try:
                symbol = dec.read(cumul, vocab_size)
            except EOFError:
                break

            decoded_tokens.append(symbol)

            if symbol == self.tokenizer.eos_token_id:
                break

            if len(decoded_tokens) % 10 == 0:
                print(f"Decoded {len(decoded_tokens)} tokens (cache: {cache_len})", end='\r')

        bitin.close()
        file_in.close()

        # Strip EOS/BOS tokens before decoding
        if decoded_tokens and decoded_tokens[-1] == self.tokenizer.eos_token_id:
            decoded_tokens.pop()
        if decoded_tokens and decoded_tokens[0] == self.tokenizer.bos_token_id:
            decoded_tokens.pop(0)

        return self.tokenizer.decode(decoded_tokens)
