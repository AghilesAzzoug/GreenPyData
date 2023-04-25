import os
import torch
import torch.distributed as dist


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 2)
    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def compliant_example_random_code():
    model = ToyModel()
    for _ in range(10):
        print(model(torch.rand(1, 10)))

def compliant_example_with_ddp(model):
    setup(0, 1)
    model_ = torch.nn.parallel.DistributedDataParallel(model)
    optim = torch.optim.SGD(model_.parameters(), lr=1e-3)

    for _ in range(10):
        inputs = torch.rand(5, 10, requires_grad=True)
        targets = torch.randint(0, 2, (5,), dtype=torch.int64)

        preds = model_(inputs)
        loss = torch.nn.functional.cross_entropy(preds, targets)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

def compliant_example_tricky(model):
    from torch.nn.parallel import DistributedDataParallel as DataParallel
    setup(1, 2)
    model_ = DataParallel(model)
    print(model_(torch.rand(5, 10)))

def non_compliant_example_fully_qualified(model):
    setup(0, 1)
    model_ = torch.nn.DataParallel(model) # Noncompliant {{Use DistributedDataParallel instead of DataParallel.}}
    print(model_(torch.rand(5, 10)))

def non_compliant_example_imported_from(model):
    from torch.nn import DataParallel
    model_ = DataParallel(model) # Noncompliant {{Use DistributedDataParallel instead of DataParallel.}}
    print(model_(torch.rand(5, 10)))

def non_compliant_example_imported_as(model):
    from torch.nn import DataParallel as DP
    model_ = DP(model) # Noncompliant {{Use DistributedDataParallel instead of DataParallel.}}
    print(model_(torch.rand(5, 10)))


