import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from openai import OpenAI
import gradio as gr
import os
from google.colab import userdata
URL = "https://drive.google.com/uc?export=download&id=13DSc38TKZoiO2C9Pby02gn5-DVQT6gX9"

# !wget -O faiss_db.zip "$URL"
# !unzip faiss_db.zip

class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

embedding_model = CustomE5Embedding(model_name="intfloat/multilingual-e5-small")
db = FAISS.load_local("faiss_db", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
api_key = userdata.get('Groq') # api_key = your Groq api key
os.environ["OPENAI_API_KEY"] = api_key
model = "llama3-70b-8192"
base_url="https://api.groq.com/openai/v1"
client = OpenAI(
    base_url=base_url # 使用 OpenAI 本身不需要這段
)
system_prompt = "你是一位臺灣的英雄聯盟的角色專家，熟悉所有英雄的技能、特性和玩法。請根據提供的資料來回應玩家的問題。回答要生動有趣，帶有英雄聯盟的風格，同時提供實用的遊戲建議。請用台灣地區常見的英雄聯盟術語回應。使用[繁體中文]回答。"

prompt_template = """
根據下列英雄資料回答問題：
{retrieved_chunks}

召喚師的問題是：{question}

請根據資料內容回覆，若資料中沒有足夠信息，請告訴召喚師可以查閱英雄聯盟官方網站或諮詢高端玩家獲取更多建議。回答時可適當融入英雄的經典台詞或遊戲中的專業術語。使用"繁體中文"回答。
"""
chat_history = []

def chat_with_rag(user_input):
    global chat_history
    # 取回相關資料
    docs = retriever.get_relevant_documents(user_input)
    retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])

    # 將自定 prompt 套入格式
    final_prompt = prompt_template.format(retrieved_chunks=retrieved_chunks, question=user_input)

    # 呼叫 OpenAI API
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt},
    ]
    )
    answer = response.choices[0].message.content

    chat_history.append((user_input, answer))
    return answer

with gr.Blocks() as demo:
    gr.Markdown("# 🎓 英雄聯盟的角色專家")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="請輸入你的問題...")

    def respond(message, chat_history_local):
        response = chat_with_rag(message)
        chat_history_local.append((message, response))
        return "", chat_history_local

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(debug=True)
