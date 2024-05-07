import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import boto3
from models.kan_former import KANFormer
import logging

# Assuming decorators and train function are defined as above

class SampleDataset(Dataset):
    """Dummy dataset for illustration purposes."""
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return torch.randn(512), torch.randint(0, 2, (512,))

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    setup_logging()
    
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KANFormer().to(device)
    logging.info("Model initialized and moved to device.")

    # Set up the data loader
    dataset = SampleDataset()
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    logging.info("Data loader setup complete.")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logging.info("Optimizer and loss function are set.")

    # Train the model
    try:
        train(model, train_loader, optimizer, criterion)
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return

    # Save the model
    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
