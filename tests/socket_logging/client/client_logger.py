import logging
import logging.handlers
import numpy as np
import socket
import traceback

logger = logging.getLogger('carla_client')
logger.setLevel(logging.INFO)

# Set up socket handler to send logs to the server
server_address = ('localhost', 9999)  # Server address and port
socket_handler = logging.handlers.SocketHandler(*server_address)
logger.addHandler(socket_handler)

frame_id = 0
episode_id = 0

try:
    while True:
        # Generate log messages with frame and episode identification
        frame_id += 1

        log_data = {
            'frame': frame_id,
            'episode': episode_id,
            'vehicle': {
                'x': np.random.randn(),
                'y': np.random.randn(),
                'z': np.random.randn(),
                'pitch': np.random.randn(),
                'roll': np.random.randn(),
                'yaw': np.random.randn(),
                'vx': np.random.randn(),
                'vy': np.random.randn(),
                'vz': np.random.randn()
            },
            'sensors': {
                'x': np.random.randn(),
                'y': np.random.randn(),
                'z': np.random.randn(),
                'pitch': np.random.randn(),
                'roll': np.random.randn(),
                'yaw': np.random.randn(),
                'vx': np.random.randn(),
                'vy': np.random.randn(),
                'vz': np.random.randn()
            }
        }
        logger.info(log_data)

        if frame_id % 100 == 0:
            episode_id += 1
            frame_id = 0
        if episode_id == 10:
            break

except KeyboardInterrupt:
    print("Keyboard Interrupt occurred.")

except Exception as e:
    print("Exception occurred: {}".format(e))
    traceback.print_exc()
finally:
    socket_handler.close()

