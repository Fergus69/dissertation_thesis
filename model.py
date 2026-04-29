import timm
import torch
import torch.nn as nn

def build_model():
    model = timm.create_model(
        'tf_efficientnetv2_s', 
        pretrained=True,       
        num_classes=2          
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
    
    return model, criterion, optimizer, device