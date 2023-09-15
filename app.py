import chainlit as cl
import torch
from transformers import pipeline
from langchain import HuggingFacePipeline, LLMChain
from ctransformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

OPENAI_API_KEY = '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'


template = """You are a helpful assistant chatbot on Professor Matthew Caesar's personal website. Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer. Use five sentences maximum. Keep the answer as concise as possible.
            {context}
            Question: {question}
            """


# get model and tokenizer from ctransformers
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", hf=True)
tokenizer = AutoTokenizer.from_pretrained(model)

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})


def split_texts():
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import WebBaseLoader

    loader = WebBaseLoader("https://caesar.web.engr.illinois.edu/")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    all_splits = text_splitter.split_documents(data)
    return all_splits

def vectordb():
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Create the vector store
    vectordb = Chroma.from_documents(
        documents=split_texts(),
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb





@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Initialize your llm and vectordb objects
    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})
    db = vectordb()

    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, vectordb=db)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["text"]).send()

# @cl.on_message
# async def main(message: str):
#     # Your custom logic goes here...
#     db = vectordb()
#
#     QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  # Run chain
#     # QA_CHAIN_PROMPT.format(context=db, question=message)
#
#
#     # Initilaize chain
#     # Set return_source_documents to True to get the source document
#     # Set chain_type to prompt template defines
#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         retriever=db.as_retriever(),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )
#
#     result = await cl.make_async(qa_chain({"query": message}))()
#
#     # Send a response back to the user
#     await cl.Message(
#         content=result["result"],
#     ).send()




# from langchain import PromptTemplate, OpenAI, LLMChain
# import chainlit as cl
#
# template = """Question: {question}
#
# Answer: Let's think step by step."""
#
#
# @cl.on_chat_start
# def main():
#     # Instantiate the chain for that user session
#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)
#
#     # Store the chain in the user session
#     cl.user_session.set("llm_chain", llm_chain)
#
#
# @cl.on_message
# async def main(message: str):
#     # Retrieve the chain from the user session
#     llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
#
#     # Call the chain asynchronously
#     res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
#
#     # Do any post processing here
#
#     # "res" is a Dict. For this chain, we get the response by reading the "text" key.
#     # This varies from chain to chain, you should check which key to read.
#     await cl.Message(content=res["text"]).send()

