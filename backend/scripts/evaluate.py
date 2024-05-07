import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.kan_former import KANFormer
import logging

# Assuming decorators are defined as previously discussed
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SampleTestDataset(Dataset):
    """Dummy test dataset for illustration purposes."""
    def __len__(self):
        return 100  # smaller test dataset

    def __getitem__(self, idx):
        return torch.randn(512), torch.randint(0, 2, (512,))

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Assuming binary classification for simplicity
            predicted = outputs.round()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KANFormer().to(device)
    model.load_state_dict(torch.load('model.pth'))
    logging.info("Model loaded.")

    test_dataset = SampleTestDataset()
    test_loader = DataLoader(test_dataset, batch_size=10)
    criterion = nn.BCELoss()  # Adjust based on your model's output and targets

    avg_loss, accuracy = evaluate(model, test_loader, criterion, device)
    logging.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
