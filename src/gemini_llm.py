import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Please set it in your .env file.\n"
        "Copy .env.example to .env and add your API key."
    )

# Initialize the client
client = genai.Client(api_key=api_key)

# Rate limiting for free tier (15 RPM = 1 request per 4 seconds to be safe)
class RateLimiter:
    def __init__(self, calls_per_minute=12):  # Conservative for free tier
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

rate_limiter = RateLimiter(calls_per_minute=12)


def gemini_call(prompt, max_retries=3):
    """
    Calls Google Gemini API with retry logic and rate limiting.
    
    Args:
        prompt: The prompt text
        max_retries: Number of retry attempts
    
    Returns:
        API response text or "NEUTRAL" on error
    """
    for attempt in range(max_retries):
        try:
            # Rate limiting to stay within free tier
            rate_limiter.wait_if_needed()
            
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=50,  # Short responses to save quota
                )
            )
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"\nAPI Error (retry {attempt+1}/{max_retries}): {e}")
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"\nAPI Error (final): {e}")
                return "NEUTRAL"
    
    return "NEUTRAL"


def gemini_call_batch(prompts, max_workers=3, show_progress=True):
    """
    Process prompts with controlled parallelism for free tier.
    
    Args:
        prompts: List of prompt strings
        max_workers: Max concurrent requests (keep low for free tier)
        show_progress: Show progress bar
    
    Returns:
        List of responses in same order as prompts
    """
    if not prompts:
        return []
    
    results = [None] * len(prompts)
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(gemini_call, prompt): idx 
            for idx, prompt in enumerate(prompts)
        }
        
        pbar = tqdm(
            total=len(prompts), 
            desc="      API calls",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            disable=not show_progress,
            ncols=80
        )
        
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"\n      ❌ Error processing prompt {idx}: {e}")
                results[idx] = "NEUTRAL"
            
            completed += 1
            pbar.update(1)
            
            # Update estimate every 10 completions
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (len(prompts) - completed) / rate if rate > 0 else 0
                pbar.set_postfix({
                    'ETA': f'{remaining/60:.1f}m',
                    'Rate': f'{rate:.1f}/s'
                }, refresh=True)
        
        pbar.close()
    
    return results
