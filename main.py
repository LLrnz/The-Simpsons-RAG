from build_rag_chain import build
from db import catalogue



rag_chain = build(catalogue)
print(rag_chain.invoke("what happens to Bart?"))
