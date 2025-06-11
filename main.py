from build_rag_chain import build


rag_chain = build("https://simpsonscrazy.net/scripts/two-cars")
print(rag_chain.invoke("what happends to Bart?"))
