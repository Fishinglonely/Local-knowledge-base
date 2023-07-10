from os import environ
from glob import glob
from langchain.document_loaders import PyPDFLoader
from llama_index import GPTMilvusIndex
import gradio as gr

HOST = "https://in03-dc41c1bb9c9c889.api.gcp-us-west1.zillizcloud.com"  # Host in Zilliz Cloud endpoint
PORT = 443  # Port in Zilliz Cloud endpoint

USER = "1023145583@qq.com"  # Username for the cluster
PASSWORD = "082930aA~"  # Password that goes with the user

environ["OPENAI_API_KEY"] = "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"  # OpenAI API Key

docs = []
for file in glob("docs/*.pdf", recursive=True):
    loader = PyPDFLoader(file)
    docs.extend(loader.load())

index = GPTMilvusIndex.from_documents(docs, host=HOST, port=PORT, user=USER, password=PASSWORD, use_secure=True,
                                      overwrite=True)


def chatbot(input_text):
    s = index.query(input_text)
    return s


demo = gr.Interface(fn=chatbot,
                    inputs=gr.inputs.Textbox(lines=7, label="输入您的文本"),
                    outputs='text',
                    title="AI 知识库聊天机器人")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7900, share=True)
