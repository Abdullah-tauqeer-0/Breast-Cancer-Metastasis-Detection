import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CamelyonDataset
from model import BreastCancerModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BreastCancerModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    dataset = CamelyonDataset("./data/train")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(10):
        for imgs in loader:
            imgs = imgs.to(device)
            # labels would be here
            # outputs = model(imgs)
            # loss = criterion(outputs, labels)
            # ...
            pass
        print(f"Epoch {epoch} complete")

if __name__ == "__main__":
    train()
