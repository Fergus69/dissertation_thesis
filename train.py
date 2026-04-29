import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from model import build_model


from dataset import AIDetectionDataset, TTADetectionDataset, train_transform, val_transform
from evaluate import evaluate_model, evaluate_model_tta, get_optimal_threshold, visualize_errors
import matplotlib.pyplot as plt
import cv2
import numpy as np

cv2.setNumThreads(0)  
torch.backends.cudnn.benchmark = True

def get_real_data(dataset_path, weights=(0.70, 0.15, 0.15)):
    train_w, val_w, test_w = weights
    
    
    assert abs(train_w + val_w + test_w - 1.0) < 1e-5, "Weights must sum to 1.0"

    all_paths = []
    all_labels = []

    
    classes = {'real': 0, 'ai': 1}

    
    for class_name, label in classes.items():
        class_dir = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Can't find directory: {class_dir}. Check the paths!")
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                all_paths.append(os.path.join(class_dir, filename))
                all_labels.append(label)

    val_test_ratio = val_w + test_w
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, 
        test_size=val_test_ratio, 
        random_state=42,
        stratify=all_labels
    )


    test_ratio_relative = test_w / val_test_ratio 

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=test_ratio_relative,
        random_state=42,
        stratify=temp_labels
    )

    print(f"Date încărcate cu succes din '{dataset_path}':")
    print(f" - Train: {len(train_paths)} imagini")
    print(f" - Validation: {len(val_paths)} imagini")
    print(f" - Test: {len(test_paths)} imagini\n")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def save_learning_curves(history):
    plt.figure(figsize=(12, 5))
    
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Accuracy Evolution (Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    print("\n[Grafic] 'learning_curves.png' a fost salvat.")

def main():
    
    model, criterion, optimizer, device = build_model()

    print(f"Training on: {device}")
    DATASET_DIR = "D:\\dataset"

    train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels = get_real_data(DATASET_DIR)

    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    
    weights = total_samples / (2.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    print(f"Calculated class weights: Real={weights[0]:.4f}, AI={weights[1]:.4f}")
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    train_dataset = AIDetectionDataset(train_image_paths, train_labels, transform=train_transform)
    val_dataset = AIDetectionDataset(val_image_paths, val_labels, transform=val_transform)
    test_dataset = AIDetectionDataset(test_image_paths, test_labels, transform=val_transform)

    batch_size = 128
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=6,
        persistent_workers=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=6,
        persistent_workers=True,
        pin_memory=True
    )
    
    tta_test_dataset = TTADetectionDataset(
        test_image_paths, 
        test_labels, 
        crop_size=224
    )
    tta_test_loader = DataLoader(
        tta_test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=6,
        persistent_workers=True,
        pin_memory=True
    )

    num_epochs = 50
    patience = 7
    patience_counter = 0
    best_val_loss = float('inf')
    save_path = 'best_model.pth'
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler()
    
    
    history = {
    'train_loss': [],
    'val_loss': [],
    'val_acc': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad() 
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        model.eval()
        running_val_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_preds += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = 100.0 * correct_preds / total_samples
        
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoca {epoch+1:02d}/{num_epochs} | '
              f'LR: {current_lr:.6f} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Val Acc: {val_accuracy:.2f}%')
        
        scheduler.step()
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_accuracy)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'   -> Saved the model! (Val Loss dropped to {best_val_loss:.4f})')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'   -> No improvement. Patience: {patience_counter}/{patience}')
            
        if patience_counter >= patience:
            print(f"\n[!] Early Stopping triggered at epoch {epoch+1}. Model has reached its maximum potential.")
            break

    save_learning_curves(history)
    print("\nTraining completed! Running detailed evaluation...")
    
    tta=1
    model.load_state_dict(torch.load(save_path))
    best_thr = get_optimal_threshold(model, val_loader, device)
    
    if tta==0:
        evaluate_model(model, test_loader, device, threshold=best_thr)
        visualize_errors(model, test_loader, device, threshold=best_thr, max_images=5)
    else:
        evaluate_model_tta(model, tta_test_loader, device, threshold=0.5)

if __name__ == '__main__':
    main()