import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from model import build_model


from dataset import AIDetectionDataset, train_transform, val_transform
from evaluate import evaluate_model

def get_real_data(dataset_path, weights=(0.70, 0.15, 0.15)):
    """
    Citește imaginile dintr-un director structurat pe clase ('real' și 'ai')
    și le împarte în Train, Val și Test conform standardului 70-15-15.
    """
    train_w, val_w, test_w = weights
    
    
    assert abs(train_w + val_w + test_w - 1.0) < 1e-5, "Ponderile trebuie să însumeze exact 1.0"

    all_paths = []
    all_labels = []

    
    classes = {'real': 0, 'ai': 1}

    
    for class_name, label in classes.items():
        class_dir = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Nu găsesc folderul: {class_dir}. Verifică structura!")
            
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

def main():
    
    model, criterion, optimizer, device = build_model()
    print(f"Antrenăm pe: {device}")
    DATASET_DIR = "E:\\disertatie\\dataset"
    
    train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels = get_real_data(DATASET_DIR)

    
    train_dataset = AIDetectionDataset(train_image_paths, train_labels, transform=train_transform)
    val_dataset = AIDetectionDataset(val_image_paths, val_labels, transform=val_transform)
    test_dataset = AIDetectionDataset(test_image_paths, test_labels, transform=val_transform)

    
    batch_size = 32 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )

    
    num_epochs = 10
    best_val_loss = float('inf')
    save_path = 'best_model.pth'

    
    for epoch in range(num_epochs):
        model.train() 
        running_train_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() 
            
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        
        model.eval() 
        running_val_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        with torch.no_grad(): 
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                
                
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_preds += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = 100.0 * correct_preds / total_samples
        
        print(f'Epoca {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Val Acc: {val_accuracy:.2f}%')
        
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'   -> Model salvat! (Val Loss a scăzut la {best_val_loss:.4f})')

    print("\nAntrenament finalizat! Se rulează evaluarea detaliată...")

    
    
    model.load_state_dict(torch.load(save_path))
    
    
    evaluate_model(model, val_loader, device)

if __name__ == '__main__':
    main()