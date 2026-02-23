from openrouter import OpenRouter
import inspect

print("OpenRouter init:", inspect.signature(OpenRouter.__init__))
try:
    client = OpenRouter(api_key="test")
    print("create:", inspect.signature(client.chat.completions.create))
except Exception as e:
    print("Error:", e)
