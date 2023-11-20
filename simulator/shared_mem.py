import sysv_ipc as ipc
import struct
import pickle


class SharedMemory:
    def __init__(self):
        path = "/tmp"

        key_flag = ipc.ftok(path, 2333)
        key_done_flag = ipc.ftok(path, 2444)

        key_recv_action = ipc.ftok(path, 3222)
        key_send_log = ipc.ftok(path, 3333)

        self.shm_flag = ipc.SharedMemory(key_flag, flags=ipc.IPC_CREAT, size=100)
        self.shm_done_flag = ipc.SharedMemory(
            key_done_flag, flags=ipc.IPC_CREAT, size=100
        )
        self.shm_recv_action = ipc.SharedMemory(
            key_recv_action, flags=ipc.IPC_CREAT, size=100
        )
        self.shm_send_log = ipc.SharedMemory(
            key_send_log, flags=ipc.IPC_CREAT, size=5000 * 8
        )

        self.shm_flag.attach(0, 0)
        self.shm_done_flag.attach(0, 0)
        self.shm_recv_action.attach(0, 0)
        self.shm_send_log.attach(0, 0)

        self.shm_objects = [
            self.shm_flag,
            self.shm_done_flag,
            self.shm_recv_action,
            self.shm_send_log,
        ]

        data_send = struct.pack("@i", 0)
        self.shm_flag.write(data_send)
        self.shm_done_flag.write(data_send)

    def write_flag(self, flag):
        data_send = struct.pack("@i", flag)
        self.shm_flag.write(data_send)

        return 0

    def read_flag(self):
        buf = self.shm_flag.read(4)
        data_recv = struct.unpack("@i", buf)

        return data_recv[0]

    def read_done_flag(self):
        buf = self.shm_done_flag.read(4)
        data_recv = struct.unpack("@i", buf)

        return data_recv[0]

    def write_done_flag(self):
        data_send = struct.pack("@i", 0)
        self.shm_done_flag.write(data_send)

        return 0

    def recieve_action_from_Gym(self):
        buf = self.shm_recv_action.read(8)
        data_recv = struct.unpack("@d", buf)
        return data_recv

    def send_log_to_Gym(self, log):
        byt1 = pickle.dumps(log)
        self.shm_send_log.write(byt1)

        return 0

    def clean(self):
        for item in self.shm_objects:
            item.detach()
            item.remove()
