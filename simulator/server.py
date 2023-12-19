import asyncio
import json
import time

from rich import print

async def handle_client(reader, writer):
    print("Client connected")
    start_time = time.time()
    received_data = b""  # Initialize an empty byte string

    while True:
        data = await reader.read(4096)
        if not data:
            break
        received_data += data  # Concatenate the received chunks

    if received_data:
        received_str = received_data.decode()  # Decode bytes to string
        received_dict = json.loads(received_str)  # Deserialize JSON string to dictionary
        # Print or process the modified dictionary
        print("Received and manipulated dictionary from client:")
        print(received_dict)
    end_time = time.time()
    print("Closing connection")
    print(f"Time taken: {end_time - start_time} s")
        
async def main():
    server = await asyncio.start_server(
        handle_client, '127.0.0.1', 8888)  # Change IP and port as needed

    addr = server.sockets[0].getsockname()
    print(f'Server running on {addr}')

    async with server:
        await server.serve_forever()

asyncio.run(main())
