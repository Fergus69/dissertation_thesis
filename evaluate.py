import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_auc_score, 
    classification_report, 
    precision_recall_curve, 
    roc_curve
)

def evaluate_model(model, loader, device):
    """
    Evaluează modelul, optimizează pragul de decizie și salvează graficele de performanță.
    Se recomandă folosirea pe test_loader la finalul antrenamentului.
    """
    model.eval()
    all_probs = []
    all_labels = []

    print("\n" + "="*30)
    print("ÎNCEPE EVALUAREA FINALĂ")
    print("="*30)
    print("Se extrag predicțiile din setul de date...")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
            
            
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    
    
    
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    
    best_idx = np.argmax(f1_scores)
    
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
    print(f"\n[Optimizare] Prag optim (Threshold) găsit: {best_threshold:.4f}")
    print(f"[Optimizare] Cel mai bun F1 Score: {f1_scores[best_idx]:.4f}")

    
    optimized_preds = (all_probs >= best_threshold).astype(int)

    
    
    
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print("\nRaport de Clasificare (folosind pragul optim):")
    report = classification_report(all_labels, optimized_preds, target_names=['Real', 'AI'])
    print(report)

    
    
    
    print("Se generează graficele de performanță...")
    
    plt.figure(figsize=(20, 6))
    
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    
    plt.subplot(1, 3, 2)
    plt.plot(recalls, precisions, color='blue', lw=2, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(alpha=0.3)

    
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(all_labels, optimized_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
    plt.title(f'Confusion Matrix (Threshold: {best_threshold:.2f})')
    plt.xlabel('Predicție Model')
    plt.ylabel('Clasă Reală')

    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300) 
    plt.show()

    print("\n[Succes] Graficul 'model_performance.png' a fost salvat.")
    print("="*30 + "\n")