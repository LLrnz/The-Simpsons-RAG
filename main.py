from build_rag_chain import create_retriever
from build_rag_chain import build
from db import catalogue


retriever = create_retriever(catalogue)
rag_chain = build(retriever)
print(rag_chain.invoke("what happens to Bart?"))
