from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
import os
import gradio as gr
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from translate import Translator
from PyPDF2 import PdfWriter, PdfReader

os.environ["OPENAI_API_KEY"] = 'sk-xxxx'

# 将给定文件夹（路径为path）下所有的pdf文档合并为一个文档
name_list = os.listdir("docs")
pdf_name_list = []
for name in name_list:
    if name[-4:] == ".pdf":
        pdf_name_list.append(name)
    else:
        pass
outputfile = open("docs" + "/" + "file.pdf", 'wb')
save_file = PdfWriter()
for name_i in pdf_name_list:
    file_i = PdfReader(open("docs" + "/" + name_i, 'rb'))
    for pageNum in range(len(file_i.pages)):
        page = file_i.pages[pageNum]
        save_file.addPage(page)
save_file.write(outputfile)
outputfile.close()

loader = PyPDFLoader('docs/file.pdf')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={
        "uri": "xxx",
        "user": "xxx",
        "password": "xxx",
        'token': "xxx",
        "secure": True
    }
)


def chatbot(input_text):
    docs = vector_db.similarity_search(input_text)
    chain = load_qa_with_sources_chain(OpenAI(temperature=0, model_name="text-davinci-003"), chain_type="map_reduce",
                                       return_intermediate_steps=True)

    dic = chain({"input_documents": docs, "question": input_text}, return_only_outputs=True)

    return dic.get("output_text", 0)


demo = gr.Interface(fn=chatbot,
                    inputs=gr.inputs.Textbox(lines=7, label="输入您的文本"),
                    outputs='text',
                    title="AI 知识库聊天机器人")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7800, share=True)
