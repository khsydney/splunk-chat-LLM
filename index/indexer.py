import os
from .loader import load_docs
from .chunker import chunk
from .embed import make_embeddings

from langchain_milvus import Milvus
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility, connections


# def build_index():
#     embs = make_embeddings()
#     docs = chunk(load_docs())

#     db_choice = os.getenv("VECTOR_DB", "milvus").lower()
#     if db_choice == "chroma":
#         from langchain_community.vectorstores import Chroma
#         vs = Chroma.from_documents(docs, embs, persist_directory="store/chroma")
#         vs.persist()
#     else:
#         from langchain_milvus import Milvus  # ✅ correct package
#         vs = Milvus.from_documents(
#             docs,
#             embs,
#             connection_args={
#                 "uri": f"http://{os.getenv('MILVUS_HOST','localhost')}:{os.getenv('MILVUS_PORT','19530')}"
#             },
#             collection_name="rag_chunks",
#         )
#     print("Index built.")

# index/indexer.py (excerpt)
def _norm_meta(md: dict) -> dict:
    return {
        "producer":       str(md.get("producer", "")),
        "creator":        str(md.get("creator", "")),
        "creationdate":   str(md.get("creationdate") or md.get("CreationDate") or ""),
        "moddate":        str(md.get("moddate") or md.get("ModDate") or ""),
        "source":         str(md.get("source") or md.get("file_path") or md.get("path") or ""),
        "total_pages":    int(md.get("total_pages") or md.get("pages") or 0),
        "page":           int(md.get("page") or 0),
        "page_label":     str(md.get("page_label") or ""),
    }

def build_index():
    # 1. Connect to Milvus FIRST
    try:
        connections.connect("default", uri="http://localhost:19530")
        if not connections.has_connection("default"):
            print("Error: Could not connect to Milvus.")
            return
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        return
    
    embs = make_embeddings()  # your HF embeddings (BAAI/bge-large-en-v1.5 => dim 1024)
    docs = chunk(load_docs()) # however you load/chunk
    texts = [d.page_content for d in docs]
    metas = [_norm_meta(d.metadata) for d in docs]

     # 1. Define the schema with explicit field types and dimensions
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024), # BGE-large-en-v1.5 is 1024-dim
        # Metadata fields
        FieldSchema(name="producer", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="creator", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="creationdate", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="moddate", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="total_pages", dtype=DataType.INT64),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="page_label", dtype=DataType.VARCHAR, max_length=128),
    ]
    schema = CollectionSchema(fields, "rag_chunks collection")

    # 2. Check and drop the old collection if it exists
    if utility.has_collection("rag_chunks"):
        utility.drop_collection("rag_chunks")

    # 3. Create the new collection
    collection = Collection(name="rag_chunks", schema=schema)

    # 4. Define and create the index with the correct metric type
    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(
        field_name="vector",
        index_params=index_params
    )
    collection.load()

    # 5. Load the documents using the existing collection
    vs = Milvus(
        collection_name="rag_chunks",
        embedding_function=embs,
        connection_args={"uri": "http://localhost:19530"},
        text_field="text",
        vector_field="vector",
        # auto_id=False # Since we are creating the collection, we can let pymilvus handle the IDs
    )
    vs.add_texts(
        texts=texts,
        metadatas=metas,
    )
    print("Index built and data loaded.")

    # from langchain_milvus import Milvus
    # vs = Milvus.from_texts(
    #     texts=texts,
    #     embedding=embs,
    #     metadatas=metas,                    # <-- ensures all required fields are present
    #     connection_args={"uri": "http://localhost:19530"},
    #     collection_name="rag_chunks",
    #     auto_id=True,                       # Milvus will create schema with proper dim
    # )
    # print("done")


if __name__ == "__main__":
    build_index()
