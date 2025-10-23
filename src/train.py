import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import PlantCNN
from dataloader import PlantVillageDataLoader
from utils import plot_training_curves, plot_confusion_matrix, get_classification_report, EarlyStopping


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader, 
                 learning_rate=0.001, num_epochs=20):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing", leave=False):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        return test_acc
    
    def train(self):
        print(f"Training for {self.num_epochs} epochs...")
        best_val_acc = 0.0
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"Saved best model (val acc: {val_acc:.2f}%)")
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
        
        # Test on best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        test_acc = self.test()
        print(f"Test accuracy: {test_acc:.2f}%")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_loader = PlantVillageDataLoader()
    loaders = data_loader.get_dataloaders(batch_size=64, num_workers=4)
    class_names = data_loader.get_class_names()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Initialize and train
    model = PlantCNN(num_classes=num_classes)
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        learning_rate=0.001,
        num_epochs=20
    )
    
    results = trainer.train()
    
    # Generate evaluation
    print("\n" + "="*60)
    print("Generating evaluation.")
    
    plot_training_curves(results, save_path='training_curves.png')
    plot_confusion_matrix(model, loaders["test"], class_names, device, 
                         save_path='confusion_matrix.png')
    get_classification_report(model, loaders["test"], class_names, device,
                             save_path='classification_report.txt')
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'class_names': class_names
    }, 'final_checkpoint.pth')
    
    print("\nSaved: best_model.pth, training_curves.png, confusion_matrix.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()