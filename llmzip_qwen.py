
import torch
import numpy as np
import bitstring
import os
import sys

# Import arithmetic coding library
# Assuming AC folder is in the same directory or python path
try:
    from AC import arithmeticcoding
except ImportError:
    # Adjust path if necessary
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from AC import arithmeticcoding

class LLMzipQwen:
    def __init__(self, model, tokenizer, device='cuda', max_window=50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_window = max_window

    def get_probs(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids)
            last_token_logits = outputs.logits[:, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)
            return probs[0].cpu().numpy()

    def encode(self, text, output_file):
        # 1. Tokenize text
        # Add BOS token implicitly or explicitly if needed. 
        # Qwen tokenizer might not add BOS by default, let's allow it to encode naturally.
        # But for reconstruction we need to be careful.
        # Let's use the tokenizer to encode the whole text first.
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        tokens = input_ids[0]
        num_tokens = len(tokens)

        print(f"Total tokens to encode: {num_tokens}")

        # 2. Initialize Arithmetic Encoder
        with open(output_file, 'wb') as f:
            bitout = arithmeticcoding.BitOutputStream(f)
            enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

            # We need a fixed context to start prediction. 
            # Standard LLMzip approach:
            # - Predict first token from BOS (or empty context if model supports it)
            # - Predict second from first, etc.
            
            # Context management
            # To predict token[i], we feed tokens[0...i-1] to the model.
            # Qwen context window is large, but for speed let's limit if needed.
            
            # Start with an empty context (or BOS if applicable)
            # Qwen doesn't always use a BOS. Let's assume we start generating from scratch.
            # If the model expects a startup, we might need a "seed". 
            # BUT, standard compression usually assumes *no* prior knowledge or a fixed seed.
            # Let's try starting with an empty input (or just BOS).
            
            # Since Qwen 2.5 is a causal LM, let's use a dummy BOS token if the list is empty?
            # Or just pass the previous tokens.
            
            # Actually, `probs = model(ids)` gives probs for the *next* token regarding the provided sequence.
            # So to encode tokens[0], we need to pass... what?
            # Usually we assume a BOS token exists at the start of the sequence for the context.
            # If tokenizer.encode adds a BOS, we can use it.
            # Qwen usually doesn't add BOS by default.
            
            # Let's fix a "BOS" convention. We can prepent a BOS token manually if needed, 
            # but then we are altering the file.
            # Better: The decoder knows it starts from nothing.
            # What does `model(tensor([], device='cuda'))` return? Maybe error.
            # Let's assume prediction of token[0] comes from a predefined BOS token (like <|endoftext|> or similar if available/appropriate).
            # Qwen's BOS token id is usually specific.
            
            # Let's use tokenizer.bos_token_id if available, or just a dummy.
            # Or we can encode the *first* token as raw bits (fixed size) to bootstrap?
            # Arithmetic coding needs a distribution. 
            # A uniform distribution over vocab for the first token is a safe fallback.
            
            # Strategy:
            # 1. Encode first token using Uniform Distribution (or learned static prior).
            #    For simplicity: Uniform distribution over vocab.
            # 2. For i = 1 to N-1:
            #    Context = tokens[0...i-1]
            #    Probs = model(Context)
            #    Encode tokens[i] using Probs
            
            # vocab_size
            vocab_size = self.model.config.vocab_size
            
            # Encode first token with uniform probabilities
            # Create a uniform cumultative frequency table
            # Frequencies: all 1.
            # Cumul: 0, 1, 2, ... vocab_size
            # This is "FlatFrequencyTable".
            # AC library expects cumul array.
            
            # Using FlatFrequencyTable equivalent logic for first token
            # But let's stick to the interface: update(cumul, symbol)
            
            # Uniform cumul
            uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
            
            # Encode first token
            first_token = tokens[0].item()
            enc.write(uniform_cumul, first_token)
            
            # Loop for the rest
            input_sequence = tokens[0:1].unsqueeze(0) # Shape [1, 1]
            
            for i in range(1, num_tokens):
                # Context management: slide window if too long
                # predicted_probs = self.get_probs(input_sequence) 
                
                # OPTIMIZATION:
                # We can use kv-cache to speed up. 
                # `outputs = model(last_token, past_key_values=...)`
                # Let's implement kv-cache usage.
                
                if i == 1:
                    outputs = self.model(input_sequence, use_cache=True)
                else:
                    # Provide only the last token and past_key_values
                    last_token_input = tokens[i-1:i].unsqueeze(0)
                    outputs = self.model(last_token_input, past_key_values=past_key_values, use_cache=True)
                
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :] # Logits for the NEXT token
                
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                # Prepare Cumulative frequencies for AC
                # We need to scale probabilities to integers summing to a power of 2 or typical AC MAX_LIMIT
                # AC library uses a specific implementation.
                # In `old_LLMzip.py`: cumul[1:] = np.cumsum(prob1*10000000 + 1)
                
                # We need to ensure strictly increasing cumulative counts.
                # Scaling factor: 10^7 is good. +1 ensures no zero-probability symbols (handling 0 prob issue).
                scale = 10000000
                freqs = (probs * scale).astype(np.uint64) + 1
                cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
                cumul[1:] = np.cumsum(freqs)
                
                # Check total doesn't exceed limit? 
                # AC.MAX_TOTAL is usually 2^30 or similar. 
                # 152k vocab * (1e7/152k + 1) ... ~1e7 + 152k. Fits easily in 32-bit or 64-bit.
                # The AC library adapts MAX_TOTAL.
                
                symbol = tokens[i].item()
                enc.write(cumul, symbol)
                
                # No need to update input_sequence for next step if using kv-cache, 
                # but valid input_sequence is needed if we weren't using kv-cache.
                # With kv-cache, we just feed the last token next time.
                
                if i % 10 == 0:
                    print(f"Encoded {i}/{num_tokens} tokens", end='\r')
            
            enc.finish()
            bitout.close()
        
        print("\nEncoding complete.")
        
        # Calculate stats
        original_size = len(text) * 8 # rough bits (assuming 1 byte chars, for ASCII) or just len(text) chars
        file_size = os.path.getsize(output_file) * 8
        bpc = file_size / len(text)
        print(f"Original size: {len(text)} chars")
        print(f"Compressed size: {os.path.getsize(output_file)} bytes")
        print(f"Bits per character: {bpc:.4f}")

    def decode(self, output_file):
        # 1. Initialize AC Decoder
        if not os.path.exists(output_file):
            print("File not found.")
            return

        file_in = open(output_file, 'rb')
        bitin = arithmeticcoding.BitInputStream(file_in)
        dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
        
        vocab_size = self.model.config.vocab_size
        
        # 2. Decode first token (Uniform)
        uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
        first_token = dec.read(uniform_cumul, vocab_size)
        
        decoded_tokens = [first_token]
        input_sequence = torch.tensor([[first_token]], device=self.device)
        past_key_values = None
        
        # 3. Decode loop
        # How do we know when to stop?
        # Usually encode length at start, or use EOS token.
        # "short_story.txt" doesn't necessarily end with EOS.
        # Simple hack: Predict fixed number, or wait for EOS, or encode length.
        # Better: Encode a special EOS token at the end of the sequence during ENCODING?
        # Or just use an EOF mechanism in AC?
        # Let's assume we decode until the file stream ends -> tricky with AC buffering.
        # Let's add a length header? Or just check against expected length if known?
        # The prompt asks for "decode natural language text", implies reproducing independent of length knowledge ideally.
        # But for this simple script, I will hardcode a limit or rely on an exception/EOS.
        # Since I am writing the scripts, I can choose to prepend the length to the file.
        
        # Re-engineering encode: Write length as first 4 bytes (int32) or just use EOS.
        # I'll modify encode slightly to prepend length in bytes? No, AC stream is bits.
        # simpler: Let's assume the user knows approximately, or let's use a "END" symbol if possible.
        # Given tokenizer, there typically is an EOS token.
        # Let's assume the text ends when we generate EOS or hit a limit?
        # But `short_story.txt` is just text.
        # I will modify `encode` to append tokenizer.eos_token_id at the end of input tokens.
        
        # Back to `decode`...
        
        # We need to close the file eventually.
        
        # Refined Plan:
        # Update `encode` to append EOS token.
        # Update `decode` to loop until EOS token is found.
        
        # Start the loop
        while True:
            # Get probs
            # Memory optimization: Limit context window size to max_window (Discards older KV cache)
            max_window = self.max_window
            current_len = len(decoded_tokens)
            
            if current_len >= max_window:
                 # Recompute window for strictly 50 tokens, discarding old cache
                 start_idx = current_len - max_window
                 window_input = torch.tensor([decoded_tokens[start_idx:]], device=self.device)
                 
                 torch.cuda.empty_cache() # Discard unnecessary kv cache
                 outputs = self.model(window_input, use_cache=True) 
                 past_key_values = outputs.past_key_values
            elif past_key_values is None:
                 outputs = self.model(input_sequence, use_cache=True)
            else:
                 last_token_input = input_sequence
                 outputs = self.model(last_token_input, past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1).detach().cpu().to(torch.float32).numpy()[0]
            
            # Sanitize probs (Same as encode)
            if np.isnan(probs).any() or np.isinf(probs).any():
                 probs = np.ones_like(probs) / len(probs)
            
            probs_sum = np.sum(probs)
            if probs_sum <= 0:
                 probs = np.ones_like(probs) / len(probs)
            else:
                 probs = probs / probs_sum

            # Construct cumul (Same as encode)
            scale = 1000000.0 # Float scale
            freqs = (probs * scale)
            freqs = np.nan_to_num(freqs, nan=1.0, posinf=1.0, neginf=1.0)
            freqs = freqs.astype(np.uint64) 
            freqs = np.maximum(freqs, 1) # Ensure at least 1
            
            cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
            cumul[1:] = np.cumsum(freqs)
            
            # Decode next symbol
            try:
                symbol = dec.read(cumul, vocab_size)
            except EOFError:
                break
                
            decoded_tokens.append(symbol)
            
            # Check for EOS
            if symbol == self.tokenizer.eos_token_id:
                break
                
            # Prepare next input
            input_sequence = torch.tensor([[symbol]], device=self.device)
            
            if len(decoded_tokens) % 1 == 0:
                print(f"Decoded {len(decoded_tokens)} tokens: {self.tokenizer.decode([symbol])}", end='\r')
                
        bitin.close()
        file_in.close()
        
        # Detokenize
        # Remove EOS if present?
        if decoded_tokens[-1] == self.tokenizer.eos_token_id:
            decoded_tokens.pop()
            
        # Remove BOS if present?
        if len(decoded_tokens) > 0 and decoded_tokens[0] == self.tokenizer.bos_token_id:
            decoded_tokens.pop(0)
            
        text = self.tokenizer.decode(decoded_tokens)
        return text

    def encode_with_eos(self, text, output_file):
        # Wrapper to ensure EOS is added
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)[0]
        # Append EOS
        eos = torch.tensor([self.tokenizer.eos_token_id], device=self.device)
        tokens = torch.cat([input_ids, eos])
        
        # Call the core logic (similar to encode above but with pre-prepared tokens)
        self._encode_tokens(tokens, output_file)

    def _encode_tokens(self, tokens, output_file):
        num_tokens = len(tokens)
        vocab_size = self.model.config.vocab_size

        with open(output_file, 'wb') as f:
            bitout = arithmeticcoding.BitOutputStream(f)
            enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
            
            # 1. First token (Uniform)
            uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
            enc.write(uniform_cumul, tokens[0].item())
            
            # 2. Loop
            input_sequence = tokens[0:1].unsqueeze(0)
            past_key_values = None
            
            for i in range(1, num_tokens):
                # Memory optimization: Limit context window size
                # If context is too long, truncate it.
                # However, with past_key_values, we can't easily truncate without recomputing.
                # Qwen supports rolling buffer?
                # Simple approach: Recompute past_key_values periodically or limit sequence length.
                # Given OOM, we MUST limit sequence length.
                # Let's use a max context window of 1024 tokens.
                
                # Memory optimization: Limit context window size to max_window tokens
                max_window = self.max_window
                
                if i >= max_window:
                     # Recompute window for strictly 50 tokens, discarding old cache
                     start_idx = i - max_window
                     window_input = tokens[start_idx:i].unsqueeze(0)
                     
                     # Clear cache before forward pass to reclaim memory
                     torch.cuda.empty_cache()
                     
                     outputs = self.model(window_input, use_cache=True) 
                     past_key_values = outputs.past_key_values
                elif past_key_values is None:
                     outputs = self.model(input_sequence, use_cache=True)
                else:
                     last_token_input = tokens[i-1:i].unsqueeze(0)
                     outputs = self.model(last_token_input, past_key_values=past_key_values, use_cache=True)
                
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1).detach().cpu().to(torch.float32).numpy()[0]
                
                # Ensure we don't exceed MAX_TOTAL of the AC implementation
                # From AC/arithmeticcoding.py: self.MAX_TOTAL = self.MIN_RANGE = (1 << 30) + 2 for 32-bit state
                # Actually, MAX_TOTAL for 32-bit state is typically 2^30.
                # Vocabulary is large (~152k). 
                # If we use scale=10^7, total is approx 10^7 + 152000, which is < 2^30 (~10^9).
                # However, float precision might be an issue.
                
                # Sanitize probs
                if np.isnan(probs).any() or np.isinf(probs).any():
                     print(f"Warning: Abnormal probabilities at token {i}")
                     # Replace NaNs/Infs with uniform
                     probs = np.ones_like(probs) / len(probs)
                
                # Normalize just in case
                probs_sum = np.sum(probs)
                if probs_sum <= 0:
                     probs = np.ones_like(probs) / len(probs)
                else:
                     probs = probs / probs_sum

                # Use a safer scale to avoid overflow
                scale = 1000000.0 # Float scale
                freqs = (probs * scale)
                freqs = np.nan_to_num(freqs, nan=1.0, posinf=1.0, neginf=1.0)
                freqs = freqs.astype(np.uint64) 
                freqs = np.maximum(freqs, 1) # Ensure at least 1
                
                # Check total
                total_freq = np.sum(freqs)
                if total_freq >= (1<<30): # Safety check against 32-bit limit
                    print(f"Warning: Frequency total {total_freq} exceeds 30-bit limit. Rescaling.")
                    freqs = (freqs / 2).astype(np.uint64)
                    freqs = np.maximum(freqs, 1)
                
                cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
                cumul[1:] = np.cumsum(freqs)
                
                enc.write(cumul, tokens[i].item())
                
                if i % 1 == 0:
                    print(f"Encoded {i}/{num_tokens} tokens", end='\r')
                    sys.stdout.flush()
                    torch.cuda.empty_cache()

            enc.finish()
            bitout.close()
