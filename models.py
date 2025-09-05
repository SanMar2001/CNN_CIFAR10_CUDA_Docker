import numpy as np
import torch
from time import time
from torch import nn
import pickle, asyncio
from torch.optim import SGD
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10

MSG_INIT = "init"
MSG_NEXT_BATCH = "next_batch"
MSG_GRADIENTS = "gradients"
MSG_EPOCH_DONE = "epoch_done"
MSG_TRAINING_DONE = "training_done"
MSG_WORKER_DONE = "worker_done"

class ParameterServer:
    def __init__(self, id, host, port, model, longitude, 
                 epochs=40, batch_size=512, lr=0.01):
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

    async def start(self):
        self.server = await asyncio.start_server(self.handle_worker, self.host, self.port)
        print(f"Parameter Server running on {self.host}:{self.port}")
        try:
            await self.server.serve_forever()
        except asyncio.CancelledError:
            print("Workers disconnected")

    def get_next_batch(self):
        start = self.batch_pointer
        end = min(start + self.batch_size, self.longitude)
        self.batch_pointer = end
        return start, end

    def reset_epoch(self):
        self.batch_pointer = 0
        self.indexes = np.random.permutation(self.longitude).tolist()
        self.current_epoch += 1

    async def handle_worker(self, reader, writer):
        addr = writer.get_extra_info("peername")
        self.active_workers.add(addr)
        print(f"Worker connected: {addr}")

        try:
            while True:
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
                        await self.send_payload(writer, {
                            "type": MSG_NEXT_BATCH,
                            "batch_range": batch_range,
                            "state_dict" : self.model.state_dict()
                        })

                    elif msg_type == MSG_GRADIENTS:
                        grads = message.get("data")
                        self.optimizer.zero_grad()
                        for param, grad in zip(self.model.parameters(), grads):
                            param.grad = grad
                        
                        self.optimizer.step()
                        
                        await self.send_payload(writer, {
                            "type": MSG_NEXT_BATCH,
                            "batch_range": self.get_next_batch(),
                            "state_dict": self.model.state_dict()
                        })

                    elif msg_type == MSG_EPOCH_DONE:
                        epoch_time = time() - self.time
                        print(f"Finished epoch {self.current_epoch} | Time: {epoch_time:.2f} s")
                        if self.current_epoch >= self.epochs:
                            await self.send_payload(writer, {"type": MSG_TRAINING_DONE})
                        else:
                            '''
                            transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.247, 0.243, 0.261])
                                    ])
                            testset = CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
                            self.model.eval()
                            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                            testloader = DataLoader(testset, batch_size=1024,
                                                    shuffle=False, num_workers=2)
                            correct = 0
                            total = 0
                            with torch.no_grad():
                                for x_batch, y_batch in testloader:
                                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                                    outputs = self.model(x_batch)
                                    preds = outputs.argmax(dim=1)
                                    correct += (preds == y_batch).sum().item()
                                    total += y_batch.size(0)
                            
                            print(f"Final accuracy in epoch: {((correct/total)*100):.2f}%\n")
                            '''
                            self.reset_epoch()
                            batch_range = self.get_next_batch()
                            self.time = time()
                            await self.send_payload(writer, {
                                "type": MSG_NEXT_BATCH,
                                "batch_range": batch_range,
                                "state_dict" : self.model.state_dict()
                            })

                    elif msg_type == MSG_WORKER_DONE:
                        self.active_workers.discard(addr)
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
        print(f"Worker device: {self.device}")

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
                if start >= len(indexes):
                    await self.send_payload(writer, {"type": MSG_EPOCH_DONE})
                    next_msg = await self.recv_payload(reader)
                    msg_type = next_msg.get("type")
                    if msg_type == MSG_TRAINING_DONE:
                        await self.send_payload(writer, {"type": MSG_WORKER_DONE})
                        print("Training finished. Closing worker")
                        return
                    else:
                        self.model.load_state_dict(next_msg["state_dict"])
                        batch_range = next_msg.get("batch_range", (len(indexes), len(indexes)))
                        if self.model is None:
                            break
                        continue

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
                await self.recv_payload(reader)

                await self.send_payload(writer, {"type": MSG_NEXT_BATCH})
                next_msg = await self.recv_payload(reader)
                # Si el server envía un state_dict, actualizar el modelo local
                if "state_dict" in next_msg:
                    self.model.load_state_dict(next_msg["state_dict"])
                    # opcional: mover el modelo otra vez al device (por seguridad)
                    self.model.to(self.device)

                batch_range = next_msg.get("batch_range", (len(indexes), len(indexes)))


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

##Definition of the model
class Convolutional_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [3,32,32] -> [32,32,32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # [32,32,32] -> [32,16,16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [32,16,16] -> [64,16,16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [64,16,16] -> [64,8,8]
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [64,8,8] -> [128,8,8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # [128,8,8] -> [128,4,4]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 1024), #Based on last features output [128,4,4]
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