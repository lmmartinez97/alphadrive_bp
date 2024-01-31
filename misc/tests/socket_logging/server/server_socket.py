from http import client
from math import pi
import socket
import struct
import pickle
import traceback
from matplotlib.backend_bases import PickEvent

from rich import print
from sklearn import dummy

def recv_msg(client_socket):

    ret = []
    index = 0
    serialized_separator = struct.pack('!2s', '@@'.encode())

    while True: #reading cycle
        return_flag = False
        size_data = b''
        serialized_msg = b''
        print("Reading size of message " + str(index))

        while True:
            byte = client_socket.recv(4) #read 4 bytes - expectedsize of message
            if serialized_separator in byte: #if separator met, then size is read
                print("Reached separator for message " + str(index))
                unpacked_size = struct.unpack('!I', size_data)[0]
                break
            else:
                size_data += byte
        
        if not size_data: #if no data, then finish reading
            return_flag = True
            break

        #if there is data, then read the message
        while len(serialized_msg) < unpacked_size:
            chunk = client_socket.recv(min(unpacked_size - len(serialized_msg), 4096))
            if not chunk:
                break
            serialized_msg += chunk
        
        dummy_bytes = 4 - (unpacked_size % 4)
        client_socket.recv(dummy_bytes)
    
        msg = pickle.loads(serialized_msg) #deserialize the message
        ret.append(msg) #add the message to the list
        index += 1
        print(msg)
        if return_flag: #if no data, then finish reading and return the list
            break

    if not ret:
        print("No data received")
        return None
    return ret
    
# Server configuration
server_address = ('localhost', 9999)  # Server address and port
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(server_address)
server_socket.listen(1)

print('Server is waiting for a connection...')
client_socket, client_address = server_socket.accept()
print('Connection from', client_address)

try:
    while True:
        if recv_msg(client_socket) is None:
            print("No data received")
            while not recv_msg(client_socket):
                pass
        else: 
            for index, received_data in enumerate(recv_msg(client_socket)):
                print("Received data " + str(index))
                print(received_data)

except KeyboardInterrupt:
    print('Interrupted by the user')

except Exception as e:
    print('Error occurred!')
    traceback.print_exc()

finally:
    # Close the client socket
    print('Closing connection...')
    client_socket.close()
