from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import torch
import torch.nn.functional as F

def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            
            
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    cm = confusion_matrix(all_labels, all_preds)
    print("Matricea de Confuzie:\n", cm)
    
    
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    
    print("\nRaport de Clasificare:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'AI']))