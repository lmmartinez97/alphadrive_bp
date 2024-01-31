from flask import Flask, render_template
import socket
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    # Function to receive data via socket
    #data = receive_data_from_socket()
    log_data = {
            'frame': 0,
            'episode': 1,
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
    return log_data  # Return received data to be displayed on the webpage

def receive_data_from_socket():
    # Create a socket to receive data (replace this with your specific socket logic)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 8888))  # Replace with your desired address and port
        s.listen(1)
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024)  # Receive data
            return data.decode('utf-8')  # Return received data as a string

if __name__ == '__main__':
    app.run(debug=True)

