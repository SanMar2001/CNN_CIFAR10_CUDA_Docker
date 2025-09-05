import argparse
import asyncio
import time
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from models import Convolutional_CIFAR10, ParameterServer, Worker

def main(args):
    if args.role == "parameter_server":
        p_server = ParameterServer(
            id = args.id,
            host = args.host,
            port = args.port,
            model = args.model,
            longitude = args.longitude
        )
        asyncio.run(p_server.start())
    
    elif args.role == "worker":
        worker = Worker(
            dataset = args.dataset,
            model = args.classmodel,
            host = args.host,
            port = args.port
        )
        time.sleep(3)
        asyncio.run(worker.run())

if __name__ == "__main__":
    #Creating parser to receive information from console
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument("--role", choices=["parameter_server", "worker"],
                        required=True)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    #Defining transforms and normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261])
    ])
    #Loading Dataset CIFAR10
    trainset = CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    longitude = len(trainset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Convolutional_CIFAR10(num_classes=10) #Classification of all the classes
    model.to(device)
    parser.set_defaults(model=model, classmodel=Convolutional_CIFAR10,
                        dataset=trainset, longitude=longitude)
    
    args = parser.parse_args()
    main(args)
    if args.role == "parameter_server":
        model.eval()

        testloader = DataLoader(testset, batch_size=1024,
                                                shuffle=False, num_workers=2)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in testloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        
        print(f"Final accuracy in test: {((correct/total)*100):.2f}%")