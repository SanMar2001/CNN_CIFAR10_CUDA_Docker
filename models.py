import numpy as np
import torch
from time import time
from torch import nn
import pickle, asyncio
from torch.optim import SGD
from torch.utils.data import Subset, DataLoader
import os

MSG_INIT = "init"
MSG_NEXT_BATCH = "next_batch"
MSG_GRADIENTS = "gradients"
MSG_TRAINING_DONE = "training_done"
MSG_WORKER_DONE = "worker_done"

class ParameterServer:
    def __init__(self, id, host, port, model, longitude, 
                 epochs=50, batch_size=2048, lr=0.004):
        self.id = id
        self.host = host
        self.port = port
        self.model = model
        self.longitude = longitude
        self.indexes = np.random.permutation(longitude).tolist()
        self.batch_size = batch_size
        self.batch_pointer = 0
        self.optimizer = SGD(self.model.parameters(), lr, momentum=0.9)
        self.epochs = epochs
        self.current_epoch = 0
        self.active_workers = set()
        self.server = None
        self.time = None
        self.gradients_buffer = {}
        self.connections = {}

    async def start(self):
        self.server = await asyncio.start_server(self.handle_worker, self.host, self.port)
        os.system("clear")
        print(f"Parameter Server running on {self.host}:{self.port}")
        try:
            await self.server.serve_forever()
        except asyncio.CancelledError:
            print("Workers disconnected")

    def get_next_batch(self):
        start = self.batch_pointer
        end = min(start + self.batch_size, self.longitude)
        if start >= self.longitude:
            return None
        self.batch_pointer = end
        return start, end

    def reset_epoch(self):
        self.batch_pointer = 0
        self.indexes = np.random.permutation(self.longitude).tolist()
        self.current_epoch += 1

    async def handle_worker(self, reader, writer):
        addr = writer.get_extra_info("peername")
        self.active_workers.add(addr)
        self.connections[addr] = writer
        print(f"Worker connected: {addr}")

        try:
            init_content = {
                "type": MSG_INIT,
                "state_dict": self.model.state_dict(),
                "indexes": self.indexes,
                "batch_range": self.get_next_batch()
            }
            self.time = time()
            await self.send_payload(writer, init_content)

            while True:
                message = await self.recv_payload(reader)
                msg_type = message.get("type")

                if msg_type == MSG_NEXT_BATCH:
                    batch_range = self.get_next_batch()
                    if batch_range is None:
                        epoch_time = time() - self.time
                        print(f"Finished epoch {self.current_epoch} | Time: {epoch_time:.2f} s")
                        if self.current_epoch >= self.epochs:
                            await self.send_payload(writer, {"type": MSG_TRAINING_DONE})
                            continue
                        self.reset_epoch()
                        self.time = time()
                        batch_range = self.get_next_batch()
                    await self.send_payload(writer, {
                        "type": MSG_NEXT_BATCH,
                        "batch_range": batch_range,
                        "state_dict": self.model.state_dict()
                    })

                elif msg_type == MSG_GRADIENTS:        
                    grads = message.get("data")
                    self.gradients_buffer[addr] = grads

                    if len(self.gradients_buffer) == len(self.active_workers) and self.active_workers:
                        self.optimizer.zero_grad()

                        for grads in self.gradients_buffer.values():
                            for param, grad in zip(self.model.parameters(), grads):
                                if param.grad is None:
                                    param.grad = grad.clone()
                                else:
                                    param.grad += grad

                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.grad /= len(self.gradients_buffer)

                        self.optimizer.step()
                        self.gradients_buffer.clear()

                        batch_range = self.get_next_batch()
                        if batch_range is None:
                            epoch_time = time() - self.time
                            print(f"Finished epoch {(self.current_epoch + 1)} | Time: {epoch_time:.2f} s")
                            if self.current_epoch >= self.epochs:
                                for w in self.connections.values():
                                    await self.send_payload(w, {"type": MSG_TRAINING_DONE})
                                return
                            self.reset_epoch()
                            self.time = time()
                            batch_range = self.get_next_batch()

                        for w in self.connections.values():
                            await self.send_payload(w, {
                                "type": MSG_NEXT_BATCH,
                                "batch_range": batch_range,
                                "state_dict": self.model.state_dict()
                            })

                elif msg_type == MSG_WORKER_DONE:
                    self.active_workers.clear()
                    return


                else:
                    print(f"Unknown message type from {addr}: {msg_type}")

        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            print(f"Worker disconnected unexpectedly: {addr}")
            self.active_workers.discard(addr)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except BrokenPipeError:
                pass
            print(f"Connection closed: {addr}")

            if not self.active_workers:
                if self.server:
                    self.server.close()
                    await self.server.wait_closed()

    async def send_payload(self, writer, obj):
        data = pickle.dumps(obj)
        writer.write(len(data).to_bytes(4, "big") + data)
        try:
            await writer.drain()
        except BrokenPipeError:
            pass

    async def recv_payload(self, reader):
        msg_len = int.from_bytes(await reader.readexactly(4), "big")
        data = await reader.readexactly(msg_len)
        return pickle.loads(data)

class Worker:
    def __init__(self, dataset, model, host, port, device=None):
        self.dataset = dataset
        self.model = model()
        self.host = host
        self.port = port
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    async def run(self):
        await asyncio.sleep(2)
        try:
            reader, writer = await asyncio.open_connection(self.host, self.port)
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return

        try:
            init_msg = await self.recv_payload(reader)
            self.model.load_state_dict(init_msg["state_dict"])
            indexes = init_msg["indexes"]
            batch_range = init_msg["batch_range"]
            loss_fn = nn.CrossEntropyLoss()

            while True:
                start, end = batch_range
                if start is None or end is None:
                    break

                batch_idx = indexes[start:end]
                subset = Subset(self.dataset, batch_idx)
                loader = DataLoader(subset, batch_size=256, shuffle=False)
                self.model.train()
                self.model.zero_grad()

                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    y_pred = self.model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    loss.backward()

                grads = [param.grad.clone() for param in self.model.parameters()]

                await self.send_payload(writer, {"type": MSG_GRADIENTS, "data": grads})
                next_msg = await self.recv_payload(reader)

                if next_msg.get("type") == MSG_TRAINING_DONE:
                    await self.send_payload(writer, {"type": MSG_WORKER_DONE})
                    print("Training finished. Closing worker")
                    return

                if "state_dict" in next_msg:
                    self.model.load_state_dict(next_msg["state_dict"])
                    self.model.to(self.device)

                batch_range = next_msg.get("batch_range", (None, None))

        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError) as e:
            print(f"Connection lost: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except BrokenPipeError:
                pass

    async def send_payload(self, writer, obj):
        data = pickle.dumps(obj)
        writer.write(len(data).to_bytes(4, "big") + data)
        await writer.drain()

    async def recv_payload(self, reader):
        msg_len = int.from_bytes(await reader.readexactly(4), "big")
        data = await reader.readexactly(msg_len)
        return pickle.loads(data)


## Modelo
class Convolutional_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
