##effcientnet

''' import flwr as fl
import torch
from collections import OrderedDict
from tqdm import tqdm

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, trainer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainer = trainer
        self.num_epochs = 20
        print(f"Training will run for {self.num_epochs} epochs")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        best_val_acc = 0
        best_model_state = None
        patience = 10
        no_improve = 0
        
        print(f"\nStarting local training for {self.num_epochs} epochs")
        
        for epoch in tqdm(range(self.num_epochs), desc='Epochs', leave=True):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_acc = self.trainer.train_epoch(self.train_loader)
            val_loss, val_acc = self.trainer.validate(self.val_loader)
            
            print(f"Training   - Loss: {train_loss:.3f}, Accuracy: {train_acc:.2f}%")
            print(f"Validation - Loss: {val_loss:.3f}, Accuracy: {val_acc:.2f}%")
            
            self.trainer.metrics['train_loss'].append(train_loss)
            self.trainer.metrics['val_loss'].append(val_loss)
            self.trainer.metrics['train_acc'].append(train_acc)
            self.trainer.metrics['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"New best model saved! Validation Accuracy: {val_acc:.2f}%")
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs")
                break

        print("\nSaving the best model...")
        torch.save(best_model_state, 'face_recognition_model.pth')
        print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        self.trainer.plot_metrics()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, val_acc = self.trainer.validate(self.val_loader)
        return val_loss, len(self.val_loader.dataset), {"accuracy": val_acc} 
         
           '''




## vGG face recognition
