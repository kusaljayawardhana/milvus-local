from pymilvus import connections, utility

connections.connect(host="milvus", port="19530")

collection_name = "healthcare_candidates"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Success: The '{collection_name}' collection has been completely wiped.")
else:
    print(f"The '{collection_name}' collection does not exist. You are good to go.")