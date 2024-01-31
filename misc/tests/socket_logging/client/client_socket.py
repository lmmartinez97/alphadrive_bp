import socket
import struct
import pickle
import time
import traceback

import numpy as np

from rich import print 

# Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 9999)  # Change to the appropriate server address and port

frame_id = 0
episode_id = 0

frame_limit = 2
episode_limit = 10

def send_handler(msg = None):
    if msg:
        serialized_data = pickle.dumps(msg)
        serialized_size = struct.pack('!I', len(serialized_data))
        serialized_separator = b'\x00\x00' + struct.pack('!2s', '@@'.encode())
        print(len(serialized_separator))

        # Send the size of the message
        print("Message size is " + str(len(serialized_data)) + ", " + str(serialized_size))
        client_socket.sendall(serialized_size)
        # Send the separator
        print("Sent size of message. Sending separator " + str(serialized_separator))
        client_socket.sendall(serialized_separator)
        # Send the message
        print(f"Sent separator - Sending message")
        client_socket.sendall(serialized_data)
        print("Sent message")

        #send blank information so that separators and messeges line in chunks of 4 bytes
        dummy_bytes = 4 - (len(serialized_data) % 4)
        print("Sending dummy bytes: " + str(dummy_bytes))
        client_socket.sendall(dummy_bytes * b'\x00')

    else:
        print("No message to send.")

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

try:
    client_socket.connect(server_address)
    # Connect to the server
    while episode_id <= episode_limit:
        send_handler(generate_log(frame_id, episode_id))
        #update frame and episode id
        frame_id += 1
        print("Frame id: " + str(frame_id) + ", Episode id: " + str(episode_id) + ". Press enter to continue.")
        if frame_id >= frame_limit:
            frame_id = 0
            print(f"Sent {frame_limit} frames of data. Episode: {episode_id}")
            episode_id += 1
        time.sleep(1)

except KeyboardInterrupt:
    print("Keyboard Interrupt occurred.")

except Exception as e:
    print("Exception occurred: {}".format(e))
    traceback.print_exc()

finally:
    # Close the socket connection
    print(f"Finished sending {episode_limit} episodes of data.")
    client_socket.close()
