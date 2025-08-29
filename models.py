import pickle

class parameterServer:
    def __init__(self, id, host, port, model):
        self.id = id
        self.host = host
        self.port = port
        self.model = model
        self.nodes = dict()
        