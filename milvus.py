from pymilvus import MilvusClient

MILVUS_URI = "http://localhost:19530"
client = MilvusClient(uri=MILVUS_URI)
collection_name = "rag_chunks"

# Let's say your vector field is named "vector"
vector_field_name = "vector"

# First, list all indexes on the collection
indexes = client.list_indexes(collection_name)
print(f"\nIndexes on collection '{collection_name}': {indexes}")

# Then, describe a specific index (e.g., the one on the vector field)
try:
    index_info = client.describe_index(
        collection_name=collection_name, 
        index_name=vector_field_name
    )
    print("\nIndex Details:")
    print(index_info)

except Exception as e:
    print(f"\nCould not describe index: {e}")
    print("This usually means no index exists on the specified field yet.")