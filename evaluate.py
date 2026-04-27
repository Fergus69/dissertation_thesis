from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, precision_recall_curve
import numpy as np
import torch
import torch.nn.functional as F

def evaluate_model(model, val_loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    print("Se extrag predicțiile pentru evaluare...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
            
            
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    
    
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"\n[Optimizare] Prag optim (Threshold) găsit: {best_threshold:.3f}")
    print(f"[Optimizare] Cel mai bun F1 Score teoretic: {f1_scores[best_idx]:.4f}")

    
    
    
    
    
    optimized_preds = (np.array(all_probs) >= best_threshold).astype(int)

    
    
    
    
    
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"\nROC AUC Score: {roc_auc:.4f}")

    
    cm = confusion_matrix(all_labels, optimized_preds)
    print("\nMatricea de Confuzie (folosind pragul optim):")
    print(cm)
    
    
    print("\nRaport de Clasificare Final:")
    print(classification_report(all_labels, optimized_preds, target_names=['Real', 'AI']))