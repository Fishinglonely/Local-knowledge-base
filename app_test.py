from os import environ
from glob import glob
import gradio as gr
from llama_index import GPTMilvusIndex
from llama_index.vector_stores import MilvusVectorStore
import MarkdownReader
from pathlib import Path

# in01-70ff1fe5d9bc5a0.aws-us-west-2.vectordb.zillizcloud.com

HOST = "xxx"  # Host in Zilliz Cloud endpoint
PORT = 443  # Port in Zilliz Cloud endpoint
TOKEN = "xxx"
# TOKEN = "xxx"
USER = "xxx"  # Username for the cluster
PASSWORD = "xxx"  # Password that goes with the user

environ["OPENAI_API_KEY"] = "sk-xxx"  # OpenAI API Key

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
