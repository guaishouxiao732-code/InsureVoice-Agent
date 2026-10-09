from api import Qwen3Client

client = Qwen3Client()

messages = [
    {"role": "system", "content": "你是一名简洁、准确的中文助手。"},
    {"role": "user", "content": "介绍一下视觉语言模型。"},
]
"""
1.编写一套合理的角色提示词
2.完成历史对话存储
3.（可以不做）压缩上下文，只能在对话过多时候触发

GPU:25G
上下文32k->如果不够，做上下文压缩
"""
for chunk in client.stream_chat(
    messages,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
):
    print(chunk, end="", flush=True)

print()