# drop_rag.py
from pymilvus import connections, utility
connections.connect("default", uri="http://localhost:19530")
if utility.has_collection("rag_chunks"):
    utility.drop_collection("rag_chunks")
print("dropped")
