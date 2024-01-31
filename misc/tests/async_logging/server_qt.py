import sys
import json
import asyncio
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit

class ServerGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Received Dictionary")
        self.setGeometry(100, 100, 600, 400)

        self.text_area = QTextEdit(self)
        self.text_area.setGeometry(10, 10, 580, 380)
        self.text_area.setReadOnly(True)

    def display_received_data(self, data):
        received_dict = json.loads(data)
        self.text_area.setText(json.dumps(received_dict, indent=4))

async def handle_client(reader, writer):
    received_data = b""
    while True:
        data = await reader.read(4096)
        if not data:
            break
        received_data += data

    if received_data:
        received_str = received_data.decode()
        server_gui.display_received_data(received_str)

async def main():
    server = await asyncio.start_server(handle_client, '127.0.0.1', 8888)
    addr = server.sockets[0].getsockname()
    print(f'Server running on {addr}')
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    server_gui = ServerGUI()

    loop = asyncio.get_event_loop()
    loop.create_task(main())

    sys.exit(app.exec_())
