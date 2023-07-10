from os import environ
from glob import glob
import gradio as gr
from llama_index import GPTMilvusIndex
from llama_index.vector_stores import MilvusVectorStore
import MarkdownReader
from pathlib import Path

# in01-70ff1fe5d9bc5a0.aws-us-west-2.vectordb.zillizcloud.com

HOST = "in03-dc41c1bb9c9c889.api.gcp-us-west1.zillizcloud.com"  # Host in Zilliz Cloud endpoint
PORT = 443  # Port in Zilliz Cloud endpoint
TOKEN = "80327f415c8bbd61d3b49e1e10ea4e989754b26659bb5b729b61111cf8a11410a477e1582c9187b6f96a5eb43d7715e5a63c445b"
# TOKEN = "1023145583@qq.com:082930aA~"
USER = "1023145583@qq.com"  # Username for the cluster
PASSWORD = "082930aA~"  # Password that goes with the user

environ["OPENAI_API_KEY"] = "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"  # OpenAI API Key

markdownreader = MarkdownReader.MarkdownReader()
# Grab all markdown files and convert them using the reader
docs = []
for file in glob("./docs/milvus-docs/site/en/**/*.md", recursive=True):
    docs.extend(markdownreader.load_data(file=Path(file)))

# Push all markdown files into Zilliz Cloud
index = GPTMilvusIndex.from_documents(docs, host=HOST, port=PORT, user=USER, password=PASSWORD, token=TOKEN,
                                      use_secure=True)


def chatbot(input_text):
    s = index.query(input_text)
    return s


demo = gr.Interface(fn=chatbot,
                    inputs=gr.inputs.Textbox(lines=7, label="输入您的文本"),
                    outputs='text',
                    title="AI 知识库聊天机器人")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7900, share=True)
