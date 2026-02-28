import os
#  Clash 软件是 7890
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY']  = 'http://127.0.0.1:7890'

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("可用模型列表：")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")