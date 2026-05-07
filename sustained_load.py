import time
import random
from pymilvus import MilvusClient

# Configuration
TARGET_RPS = 100
DURATION_SECONDS = 60
DIMENSION = 768
COLLECTION_NAME = "load_test_collection"

print("Connecting to Milvus...")
client = MilvusClient(uri="http://localhost:19530")

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

client.create_collection(collection_name=COLLECTION_NAME, dimension=DIMENSION)

print(f"\nStarting Sustained Load Test: {TARGET_RPS} RPS for {DURATION_SECONDS} seconds...")
print("Switch to your 'docker stats' terminal to watch the CPU/RAM usage!\n")

# Pre-generate a pool of random dummy vectors to avoid CPU overhead during the test
dummy_pool = [[random.random() for _ in range(DIMENSION)] for _ in range(TARGET_RPS)]

successful_requests = 0

for current_second in range(DURATION_SECONDS):
    loop_start_time = time.time()
    
    # Prepare exactly 100 records for this second
    batch_data = [
        {"id": (current_second * TARGET_RPS) + i, "vector": dummy_pool[i]} 
        for i in range(TARGET_RPS)
    ]
    
    # Fire the batch directly into Milvus
    client.insert(collection_name=COLLECTION_NAME, data=batch_data)
    successful_requests += TARGET_RPS
    
    # Calculate how long the insertion took
    elapsed_time = time.time() - loop_start_time
    
    # Print progress every 5 seconds
    if current_second % 5 == 0:
        print(f"[{current_second}s] Sent {TARGET_RPS} requests... (Insertion took {elapsed_time:.4f}s)")
    
    # Sleep for the remainder of the second to maintain exactly 100 RPS
    sleep_time = max(0, 1.0 - elapsed_time)
    time.sleep(sleep_time)

print(f"\n✅ Load Test Complete. Total Vectors Inserted: {successful_requests}")
client.drop_collection(COLLECTION_NAME)
