import asyncio
import json
import numpy as np
import time

from rich import print

def generate_log(frame_id, episode_id):
    log_data = {
        'frame': frame_id,
        'episode': episode_id,
        'vehicle': {
            'position': {
                'x': np.random.uniform(-10, 10),
                'y': np.random.uniform(-10, 10),
                'z': np.random.uniform(-10, 10)
            },
            'rotation': {
                'x': np.random.uniform(-10, 10),
                'y': np.random.uniform(-10, 10),
                'z': np.random.uniform(-10, 10)
            },
            'velocity': {
                'x': np.random.uniform(-10, 10),
                'y': np.random.uniform(-10, 10),
                'z': np.random.uniform(-10, 10)
            },
            'acceleration': {
                'x': np.random.uniform(-10, 10),
                'y': np.random.uniform(-10, 10),
                'z': np.random.uniform(-10, 10)
            }
        }
    }

    return log_data


async def send_log_dictionary(host, port, log):
    reader, writer = await asyncio.open_connection(host, port)

    # Serialize the dictionary to JSON string and encode it to UTF-8
    data = json.dumps(log).encode()
    writer.write(data)
    await writer.drain()

    writer.close()
    await writer.wait_closed()

async def main():
    host = '127.0.0.1'  # Server IP
    port = 8888  # Server port
    episode_limit  = 2
    frame_limit = 10
    episode_id = 0
    frame_id = 0

    print("Starting to send data.")
    start_time = time.time()
    while episode_id <= episode_limit:
        while frame_id <= frame_limit:
            log = generate_log(frame_id, episode_id)
            await send_log_dictionary(host, port, log)
            print("Frame id: " + str(frame_id) + ", Episode id: " + str(episode_id) + ".")
            frame_id += 1
            episode_id += 1
        print(f"Sent {frame_limit} frames of data. Episode: {episode_id}")
        frame_id = 0
        episode_id += 1
    end_time = time.time()
    print(f"Finished sending {episode_limit} episodes of data.")
    print(f"Total time: {end_time - start_time} s")

asyncio.run(main())

