from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate


system_prompt = (
    "You are a Medical assistant that helps users with their medical questions." 
    "Use the following retrieved documents to answer the user's question." 
    "If you don't know the answer, say you don't know."
    "Use three sentences at most and be concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", system_prompt),
    ("human", "{input}"),
  ]
)