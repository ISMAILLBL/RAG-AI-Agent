from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_3Zw4Jk_E5Cg8PKPzxs4KqabfGJwfxvi8E6eDwEUwU5vXBwPEdCBckm4aiEhGVLCeMePExP")
index = pc.Index("rag-demo")   # ton index

index.delete(delete_all=True)

print("Tous les embeddings ont été supprimés !")
