# from gpt_index import SimpleDirectoryReader, GPTListIndex,readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import asyncio
from types import FunctionType
from llama_index import ServiceContext, GPTVectorStoreIndex, LLMPredictor, PromptHelper, SimpleDirectoryReader, \
    load_index_from_storage
import sys
import os
import time
# from llama_index.response.schema import StreamingResponse
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

os.environ["OPENAI_API_KEY"] = "your key here"  # gpt 3.5 turbo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from llama_index import StorageContext, load_index_from_storage, ServiceContext
from langchain.chat_models import ChatOpenAI


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 5000
    max_chunk_overlap = 256
    chunk_size_limit = 3900

    print("*" * 5, "Documents parsing initiated", "*" * 5)
    file_metadata = lambda x: {"filename": x}
    reader = SimpleDirectoryReader(directory_path, file_metadata=file_metadata)
    print(reader)
    documents = reader.load_data()
    print("*" * 5, "Documents parsing done", "*" * 5)

    print(documents[0].extra_info)
    print(documents[0].doc_id)

    print()
    # nodes = parser.get_nodes_from_documents(documents)
    # index = GPTVectorStoreIndex(nodes)
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTVectorStoreIndex.from_documents(
        documents=documents, service_context=service_context
    )

    index.storage_context.persist("./jsons/contentstack_llm")
    return index


def get_index():
    max_input_size = 4000
    num_outputs = 1024
    max_chunk_overlap = 512
    chunk_size_limit = 3900
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs, streaming=True))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    return service_context


# construct_index("./documents")
#
storage_context = StorageContext.from_defaults(persist_dir="./your_directory")

service_context = get_index()
index = load_index_from_storage(storage_context, service_context=service_context)

query_engine = index.as_query_engine(streaming=True)


async def astreamer(generator):
    try:
        for i in generator:
            yield (i)
            await asyncio.sleep(.1)
    except asyncio.CancelledError as e:

        print('cancelled')


class Item(BaseModel):
    input_text: str


@app.post("/question_answering")
async def create_item(item: Item):
    input_sentence = item.input_text
    response = query_engine.query(input_sentence)
    return StreamingResponse(astreamer(response.response_gen), media_type="text/event-stream")


@app.get("/")
@app.get("/health_check")
async def health_check():
    return "ok"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
