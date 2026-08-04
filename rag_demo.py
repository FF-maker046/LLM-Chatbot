import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. 加载环境变量
load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=api_key)

# 2. 加载PDF并切分
pdf_path = "data/wan_runge_kutta.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"✅ PDF已切分为 {len(chunks)} 个片段")

# 3. 构建向量索引（用本地模型，不花钱）
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)


# 4. 检索 + 问答
def ask_pdf(question):
    # 检索相关片段
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 构造提示词
    prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请直接说"文档中没有提到相关内容"。

文档内容：
{context}

问题：{question}
回答："""

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# 5. 测试
if __name__ == "__main__":
    question = "这篇论文提出了什么新方法？"
    answer = ask_pdf(question)
    print(f"问题：{question}")
    print(f"回答：{answer}")