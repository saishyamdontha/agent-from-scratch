import statistics, time, requests

URL = "http://localhost:11434/api/chat"
PROMPT = "Reply with one word: hello"

def one_call(think):
    start = time.time()
    requests.post(URL, json={"model": "qwen3:8b", "stream": False, "think": think,
                             "messages": [{"role": "user", "content": PROMPT}]})
    return time.time() - start

one_call(False)  # warm-up: make sure the model is loaded, not timed

for think in (True, False):
    times = [one_call(think) for _ in range(5)]
    print(f"think={think}: median {statistics.median(times):.1f}s, "
          f"min {min(times):.1f}s, max {max(times):.1f}s")
