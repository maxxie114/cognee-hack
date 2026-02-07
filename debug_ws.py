
import asyncio
import json
import websockets
import sys

async def test_chat():
    uri = "ws://localhost:8002/ws/chat"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Send a simple message
            msg = {"message": "Hello, is this working?"}
            await websocket.send(json.dumps(msg))
            print(f"Sent: {msg}")

            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    print(f"Received: {response}")
                    data = json.loads(response)
                    if data.get("type") == "done":
                        break
                    elif data.get("type") == "error":
                        print(f"Error from server: {data}")
                        break
                except asyncio.TimeoutError:
                    print("Timeout waiting for response")
                    break
                    
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_chat())
