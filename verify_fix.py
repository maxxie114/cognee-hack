
from qdrant_client import QdrantClient
import inspect

client = QdrantClient(url="http://localhost:6333")
print(inspect.signature(client.query_points))
print(client.query_points.__doc__)
