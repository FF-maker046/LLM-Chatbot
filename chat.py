import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 1. 加载环境变量
load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("❌ 请先配置 .env 文件中的 ZHIPUAI_API_KEY")

client = ZhipuAI(api_key=api_key)

# 2. 创建一个空的“记忆容器”，用来存放对话历史
messages = []

print("🤖 你的 AI 助手已上线（输入 'exit' 退出）")
print("-" * 30)

# 3. 开启无限循环，实现连续对话
while True:
    # 获取你的输入
    user_input = input("👤 你：")

    # 退出条件
    if user_input.lower() == "exit":
        print("👋 再见！")
        break

    # 把你的问题加入“记忆容器”
    messages.append({"role": "user", "content": user_input})

    # 调用 API
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages  # 把整个对话历史都传给 AI
        )
        reply = response.choices[0].message.content

        # 把 AI 的回答也加入“记忆容器”
        messages.append({"role": "assistant", "content": reply})

        print(f"🤖 AI：{reply}")
        print("-" * 30)
    except Exception as e:
        print(f"⚠️ 出错了：{e}")
