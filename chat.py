from api import Qwen3Client

client = Qwen3Client()

messages = client.build_messages("介绍一下视觉语言模型。")
print(client.chat(messages))