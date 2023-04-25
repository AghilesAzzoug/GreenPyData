import torch
from torch.utils.data import DataLoader
import torchvision

dl = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False) # Noncompliant {{Use asynchronous data loading for better GPU usage.}}

dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False) # Noncompliant {{Use asynchronous data loading for better GPU usage.}}

dl = torch.utils.data.DataLoader(dataset, num_workers=5, batch_size=2, shuffle=True)

dl = torch.utils.data.DataLoader(dataset, num_workers=5, shuffle=True)

dl = DataLoader(dataset, num_workers=5, shuffle=True)

dl = DataLoader(dataset, 1, False, None, None, 0, None, False, False) # Noncompliant {{Use asynchronous data loading for better GPU usage.}}

dl = DataLoader(dataset, 1, False, None, None, 0) # Noncompliant {{Use asynchronous data loading for better GPU usage.}}

dl = DataLoader(dataset, 1, False, None, None, num_workers=0) # Noncompliant {{Use asynchronous data loading for better GPU usage.}}

dl = DataLoader(dataset, 1, False, None, None, 10, None, False, False)

dl = DataLoader(dataset) # Noncompliant {{Use asynchronous data loading for better GPU usage.}}

dl = DataLoader(dataset, batch_size=1,) # Noncompliant {{Use asynchronous data loading for better GPU usage.}}

dl = torch.utils.data.DataLoader(dataset, batch_size=1) # Noncompliant {{Use asynchronous data loading for better GPU usage.}}

my_dataloader = torch.utils.data.DataLoader
dl_class = DataLoader
