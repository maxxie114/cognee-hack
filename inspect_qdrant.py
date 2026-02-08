
from qdrant_client import QdrantClient
import inspect

client = QdrantClient(url="http://localhost:6333")
print(f"Client type: {type(client)}")
print(f"Has search: {hasattr(client, 'search')}")
print(f"Has query: {hasattr(client, 'query')}")
print(f"Dir: {dir(client)}")
