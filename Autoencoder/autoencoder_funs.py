import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import torch.optim as optim

def show_dataloader_batch(dataloader, num_images=6):
    images, labels = next(iter(dataloader))
    
    images = images[:num_images]
    labels = labels[:num_images]
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    
    for i, ax in enumerate(axes):
        img = images[i].numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1) 
        
        ax.imshow(img)
        
        class_name = "Anomaly" if labels[i].item() == 1 else "Normal"
        
        color = 'red' if labels[i].item() == 1 else 'black'
        ax.set_title(class_name, color=color, fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_training_loss(epoch_losses, criterion):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss', color='#1f77b4', linewidth=2.5)

    plt.title('Autoencoder Reconstruction Loss over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)

    if criterion.__class__.__name__ == 'MSELoss':
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    elif criterion.__class__.__name__ == 'L1Loss':
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def compare_training_losses(losses_dict):
    plt.figure(figsize=(10, 5))
    
    for model_name, losses in losses_dict.items():
        plt.plot(losses, label=model_name, linewidth=2.5)

    plt.title('Training Loss Comparison of Autoencoder Models', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def train_autoencoder(model, train_loader, criterion, lr, epochs,device):
    model = model.to(device)
    EPOCHS = epochs
    LEARNING_RATE = lr

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_losses = []

    print("Starting Training...")

    for epoch in range(EPOCHS):
        
        model.train() 
        epoch_loss = 0.0
        
        for images, _ in train_loader: 
            images = images.to(device)
            
            reconstructions = model(images)
            
            loss = criterion(reconstructions, images)
            
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      

            epoch_loss += loss.item()
            
        epoch_losses.append(epoch_loss/len(train_loader)) # average loss per batch for the epoch
        
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {epoch_losses[-1]:.4f}")

    print("\n Training Complete!")
    return epoch_losses

def evaluate_autoencoder(model, val_loader, criterion, device):
    model.eval()
    if criterion.__class__.__name__ == 'MSELoss':
        criterion = nn.MSELoss(reduction='none')
    elif criterion.__class__.__name__ == 'L1Loss':
        criterion = nn.L1Loss(reduction='none')
    elif criterion.__class__.__name__ == 'BCELoss':
        criterion = nn.BCELoss(reduction='none')

    all_labels = []
    all_scores = []

    print("Running Validation Set with Top-K% Anomaly Scoring...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            reconstructions = model(images)
            
            pixel_losses = criterion(reconstructions, images)
            
            spatial_losses = pixel_losses.mean(dim=1)
            
            flat_losses = spatial_losses.view(images.size(0), -1)
            
            k_pixels = int(flat_losses.size(1) * 0.1)
            topk_losses, _ = torch.topk(flat_losses, k=k_pixels, dim=1)
            
            image_scores = topk_losses.mean(dim=1)
            # image_scores = flat_losses.mean(dim=1)
            
            all_scores.extend(image_scores.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"\n Evaluation Complete!")
    print(f"Final AUROC Score: {auroc:.4f}")

    normal_scores = all_scores[all_labels == 0]
    anomaly_scores = all_scores[all_labels == 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(normal_scores, bins=50, alpha=0.6, color='#1f77b4', label=f'Normal (n={len(normal_scores)})', density=True)
    axes[0].hist(anomaly_scores, bins=50, alpha=0.6, color='#d62728', label=f'Anomaly (n={len(anomaly_scores)})', density=True)
    axes[0].set_title('Distribution of Anomaly Scores (MSE)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    axes[1].plot(fpr, tpr, color='#ff7f0e', lw=2, label=f'ROC curve (area = {auroc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # The random-guess baseline
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Mathematical Optimal Anomaly Threshold (MSE): {optimal_threshold:.6f}\n")

    predictions = (all_scores > optimal_threshold).astype(int)

    accuracy = accuracy_score(all_labels, predictions)
    print(f"Final Model Accuracy: {accuracy * 100:.2f}%\n")

    print("--- Detailed Classification Report ---")
    print(classification_report(all_labels, predictions, target_names=['Normal', 'Anomaly']))

    cm = confusion_matrix(all_labels, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['Actual Normal', 'Actual Anomaly'],
                annot_kws={"size": 14, "weight": "bold"})

    plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.4f})', fontsize=14, fontweight='bold', pad=15)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def check_reconstructions(model, test_loader, device):
    model.eval()
    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        reconstructions = model(images)

    images = images.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Top Row: Originals | Bottom Row: Autoencoder Reconstructions", fontsize=16)

    for i in range(5):
        orig_img = np.transpose(images[i], (1, 2, 0))
        recon_img = np.transpose(reconstructions[i], (1, 2, 0))
        
        orig_img = np.clip(orig_img, 0, 1)
        recon_img = np.clip(recon_img, 0, 1)
        
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Label: {'Anomaly' if labels[i]==1 else 'Normal'}")
        
        axes[1, i].imshow(recon_img)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()