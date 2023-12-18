import socket
import logging

# Setup socket server to receive logs
server_address = ('localhost', 9999)  # Server address and port
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('socket_logger')

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(server_address)
server_socket.listen(1)

print('Waiting for a connection...')
connection, client_address = server_socket.accept()
print('Connection from', client_address)

while True:
    data = connection.recv(1024)  # Adjust buffer size as needed
    if not data:
        break
    print(data)
    logger.info(data.decode('utf-8'))  # Process log messages as needed

connection.close()
