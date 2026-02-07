import os
from dotenv import load_dotenv

# Load env vars BEFORE any cognee imports
load_dotenv()
os.environ.setdefault("VECTOR_DB_PROVIDER", "qdrant")
os.environ.setdefault("VECTOR_DB_URL", os.getenv("QDRANT_URL", ""))
os.environ.setdefault("VECTOR_DB_KEY", os.getenv("QDRANT_API_KEY", ""))

# Register Qdrant adapter BEFORE importing cognee
try:
    import cognee_community_vector_adapter_qdrant.register  # noqa: F401
except ImportError:
    pass

import cognee
import asyncio
from helper_functions import import_cognee_data
from cognee.api.v1.visualize.visualize import visualize_graph


async def main():
    # Clear local cognee data (graph + metadata), but NOT vectors (they're on Qdrant Cloud)
    print("Clearing local cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(graph=True, vector=False, metadata=True)
    print("✓ Local data cleared (Qdrant Cloud vectors preserved)\n")

    # Import everything (graph, vector, system DB, data storage)
    print("Importing data from export...")
    success = await import_cognee_data("cognee_export", verbose=True)
    
    if not success:
        print("\n✗ Import failed!")
        return
    
    # Create visualization
    print("\nCreating graph visualization...")
    await visualize_graph("./graphs/after_setup.html")
    
    print(f"\n{'='*60}")
    print(f"✓ Import verification complete!")
    print(f"  Graph visualization: ./graphs/after_setup.html")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())