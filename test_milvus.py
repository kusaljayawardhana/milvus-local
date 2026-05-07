import time
from pymilvus import MilvusClient

print("Connecting to Milvus...")
client = MilvusClient(uri="http://localhost:19530")

collection_name = "test_cv_vectors"
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

client.create_collection(collection_name=collection_name, dimension=3)

print("Inserting fake CV data...")
data = [
    {"id": 1, "vector": [0.1, 0.2, 0.3], "name": "Candidate A (A&E Nurse)"},
    {"id": 2, "vector": [0.9, 0.8, 0.7], "name": "Candidate B (ICU Nurse)"}
]
client.insert(collection_name=collection_name, data=data)

# FIX: Give Milvus 2 seconds to index the newly inserted data
print("Waiting for database to index...")
time.sleep(2)

print("Searching for best match...")
results = client.search(
    collection_name=collection_name,
    data=[[0.8, 0.8, 0.8]], 
    limit=1,
    output_fields=["name"]
)

print(f"\nTop Match: {results[0][0]['entity']['name']}")
