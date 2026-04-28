import torch
import cv2
import os
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

def visualize_errors(model, loader, device, threshold=0.5, max_images=10):
    model.eval()
    all_errors = [] 
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print("\n[Analiză] Căutăm cele mai mari greșeli ale modelului...")

    with torch.no_grad():
        for images, labels, paths in loader:
            images_gpu = images.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images_gpu)
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            probs_cpu = probs.cpu().numpy()
            labels_cpu = labels.numpy()
            preds = (probs_cpu >= threshold).astype(int)

            mis_idx = np.where(preds != labels_cpu)[0]

            for idx in mis_idx:
                
                error_score = abs(labels_cpu[idx] - probs_cpu[idx])
                
                
                ai_crop = images[idx].permute(1, 2, 0).numpy()
                ai_crop = (std * ai_crop + mean).clip(0, 1)

                all_errors.append({
                    'score': error_score,
                    'path': paths[idx],
                    'ai_input': ai_crop,
                    'true': labels_cpu[idx],
                    'pred': preds[idx],
                    'prob': probs_cpu[idx]
                })

    if not all_errors:
        print("Nu s-au găsit erori.")
        return

    
    all_errors.sort(key=lambda x: x['score'], reverse=True)
    top_errors = all_errors[:max_images]

    
    num_to_show = len(top_errors)
    fig, axes = plt.subplots(num_to_show, 2, figsize=(14, 4 * num_to_show))
    if num_to_show == 1: axes = np.expand_dims(axes, axis=0)

    for i, err in enumerate(top_errors):
        
        full_img = cv2.cvtColor(cv2.imread(err['path']), cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(full_img)
        axes[i, 0].set_title(f"Confuzie maximă - Original: {os.path.basename(err['path'])}")
        axes[i, 0].axis('off')

        
        axes[i, 1].imshow(err['ai_input'])
        t_cls = 'AI' if err['true'] == 1 else 'Real'
        p_cls = 'AI' if err['pred'] == 1 else 'Real'
        axes[i, 1].set_title(f"T:{t_cls} | P:{p_cls} | Siguranță eronată: {err['score']*100:.1f}%", 
                             color='red', weight='bold', fontsize=12)
        axes[i, 1].axis('off')

    plt.tight_layout(pad=3.0)
    plt.savefig('worst_10_errors.png', dpi=150)
    plt.show()
    print(f"\n[Succes] Top {num_to_show} cele mai sigure greșeli salvate în 'worst_10_errors.png'.")
    
    
    
def get_optimal_threshold(model, loader, device):
    """
    Calculates the best threshold on a specific loader (e.g., Validation Set)
    to maximize the F1 Score.
    """
    model.eval()
    all_probs, all_labels = [], []
    
    print("Finding optimal threshold on the provided loader...")
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
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
    
    print(f"Optimal threshold found: {best_threshold:.4f} with F1: {f1_scores[best_idx]:.4f}")
    return best_threshold

def evaluate_model(model, loader, device, threshold=0.5):
    """
    Evaluates the model on a loader (e.g., Test Set) using a PREDEFINED threshold.
    Generates metrics and professional plots for the dissertation.
    """
    model.eval()
    all_probs = []
    all_labels = []

    print("\n" + "="*40)
    print(f"STARTING FINAL EVALUATION (Threshold: {threshold:.4f})")
    print("="*40)

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    
    optimized_preds = (all_probs >= threshold).astype(int)

    
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print(f"\nClassification Report (Fixed Threshold: {threshold:.4f}):")
    print(classification_report(all_labels, optimized_preds, target_names=['Real', 'AI']))

    
    print("Generating performance plots...")
    plt.figure(figsize=(20, 6))
    
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    
    precisions, recalls, _ = precision_recall_curve(all_labels, all_probs)
    plt.subplot(1, 3, 2)
    plt.plot(recalls, precisions, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(alpha=0.3)

    
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(all_labels, optimized_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
    plt.title(f'Confusion Matrix (Fixed Thr: {threshold:.2f})')
    plt.xlabel('Model Prediction')
    plt.ylabel('True Class')

    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300)
    plt.show()

    print("\n[Success] Evaluation completed. Plots saved as 'model_performance.png'.")