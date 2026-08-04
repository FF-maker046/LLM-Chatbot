import streamlit as st
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

# --- 页面配置 ---
st.set_page_config(page_title="AI 学术助手", page_icon="🎓", layout="centered")

# --- 加载环境变量 ---
load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    st.error("⚠️ 请先配置 .env 文件中的 ZHIPUAI_API_KEY")
    st.stop()

client = ZhipuAI(api_key=api_key)

# --- 自定义CSS（美化） ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .user-message {
        background-color: #e8f0fe;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 6px 0;
        max-width: 85%;
        margin-left: auto;
        color: #1a1a1a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .assistant-message {
        background-color: #f1f3f5;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 6px 0;
        max-width: 85%;
        color: #1a1a1a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# --- 标题 ---
st.markdown('<p class="main-title">🎓 AI 学术助手</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于智谱 GLM · 支持聊天 & PDF 问答</p>', unsafe_allow_html=True)

# --- 侧边栏：模式选择 + 设置 ---
with st.sidebar:
    st.markdown("### ⚙️ 模式设置")
    mode = st.radio(
        "选择功能模式：",
        ["💬 自由聊天", "📄 PDF 问答"],
        index=0
    )

    if mode == "💬 自由聊天":
        system_prompt = st.text_area(
            "系统提示词（控制AI角色）",
            value="你是一位严谨的计算机科学导师，回答要简洁、有条理，多用分点说明。",
            height=150
        )
    else:
        st.markdown("**上传 PDF 文档**")
        uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf")
        if uploaded_file:
            st.success(f"已上传：{uploaded_file.name}")

# --- 初始化对话历史（两种模式共用一套消息列表，但存储时加上mode标记） ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = mode

# 如果切换模式，清空对话历史（避免混淆）
if st.session_state.mode != mode:
    st.session_state.messages = []
    st.session_state.mode = mode

# --- 显示历史消息 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 处理用户输入 ---
if prompt := st.chat_input("输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 根据模式处理
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            if mode == "💬 自由聊天":
                # ---- 普通聊天模式 ----
                system_prompt_dict = {"role": "system", "content": system_prompt}
                messages_with_system = [system_prompt_dict] + st.session_state.messages
                try:
                    response = client.chat.completions.create(
                        model="glm-4-flash",
                        messages=messages_with_system
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    reply = f"❌ 出错了：{e}"
            else:
                # ---- PDF 问答模式 ----
                if not uploaded_file:
                    reply = "⚠️ 请先在侧边栏上传 PDF 文件。"
                else:
                    try:
                        # 1. 保存上传的PDF到临时文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        # 2. 加载并切分PDF
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = text_splitter.split_documents(docs)


                        # 3. 构建向量库（使用缓存机制，避免重复加载模型）
                        @st.cache_resource
                        def load_embeddings():
                            return HuggingFaceEmbeddings(
                                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                            )


                        embeddings = load_embeddings()
                        vectorstore = FAISS.from_documents(chunks, embeddings)

                        # 4. 检索相关片段
                        retrieved_docs = vectorstore.similarity_search(prompt, k=3)
                        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                        # 5. 构造提示并调用大模型
                        rag_prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请直接说"文档中没有提到相关内容"。

文档内容：
{context}

问题：{prompt}
回答："""
                        response = client.chat.completions.create(
                            model="glm-4-flash",
                            messages=[{"role": "user", "content": rag_prompt}]
                        )
                        reply = response.choices[0].message.content

                        # 删除临时文件
                        os.unlink(tmp_path)
                    except Exception as e:
                        reply = f"❌ RAG 处理失败：{e}"
                        if "uploaded_file" in locals():
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass

    # 保存AI回复
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.write(reply)