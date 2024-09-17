#load dataset and pass it through resnet18 without preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import seaborn as sns
from datasets import load_dataset
import numpy as np

# Load dataset from Hugging Face datasets
dataset = load_dataset("hf-vision/chest-xray-pneumonia")

# Access training and test splits
train_data = dataset['train']
test_data = dataset['test']

# Function to count samples and labels
def count_samples_and_labels(data):
    total_samples = len(data)
    normal_count = sum(1 for example in data if example['label'] == 0)
    pneumonia_count = sum(1 for example in data if example['label'] == 1)
    return total_samples, normal_count, pneumonia_count

# Count samples and labels in training data
train_total, train_normal, train_pneumonia = count_samples_and_labels(train_data)

# Count samples and labels in test data
test_total, test_normal, test_pneumonia = count_samples_and_labels(test_data)

# Print the results
print("Training Data:")
print("Total Samples:", train_total)
print("Normal (Label 0):", train_normal)
print("Pneumonia (Label 1):", train_pneumonia)
print()
print("Test Data:")
print("Total Samples:", test_total)
print("Normal (Label 0):", test_normal)
print("Pneumonia (Label 1):", test_pneumonia)

# Custom dataset class for Chest X-Ray images
class ChestXRayDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
      sample = self.dataset[idx]
      image = sample["image"]
      label = sample["label"]
      if self.transform:
          image = self.transform(image)
      return {"image": image, "label": label}

if __name__ == '__main__':
     # Define transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load data using custom dataset class with transformations
    train_data = ChestXRayDataset(dataset["train"], transform=transform)
    test_data = ChestXRayDataset(dataset["test"], transform=transform)

    # Define the ResNet18 model
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Define DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Train model function
    def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
        epoch_recall = []
        epoch_balanced_accuracy = []
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                images, labels = batch['image'], batch['label']
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # Evaluate the model after each epoch
            predictions, true_labels = evaluate_model(model, test_loader)
            _, recall, balanced_accuracy = calculate_metrics(true_labels, predictions)
            epoch_recall.append(recall)
            epoch_balanced_accuracy.append(balanced_accuracy)
        return epoch_recall, epoch_balanced_accuracy

    # Evaluate model function
    def evaluate_model(model, test_loader):
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch['image'], batch['label']
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.tolist())
                true_labels.extend(labels.tolist())
        return predictions, true_labels

    # Function to calculate metrics (F1 score, recall, balanced accuracy)
    def calculate_metrics(true_labels, predictions):
        f1 = f1_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)

        # Calculate specificity
        true_negatives = sum([1 for true_label, pred_label in zip(true_labels, predictions) if true_label == pred_label == 0])
        false_positives = sum([1 for true_label, pred_label in zip(true_labels, predictions) if true_label == 0 and pred_label == 1])
        specificity = true_negatives / (true_negatives + false_positives)

        # Calculate balanced accuracy
        balanced_accuracy = (recall + specificity) / 2
        return f1, recall, balanced_accuracy


    #SmoothGradCAMpp visualization function
    def smoothgradcampp_visualization(model, test_loader, model_name):
        # Freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze specific layer(s)
        for param in model.layer4.parameters():
            param.requires_grad = True
        img = next(iter(test_loader))['image'][0]
        tensor = img.unsqueeze(0)  # Move input tensor to the same device as the model
        target_layer = model.layer4[1] # the target layer you want to visualize
        with SmoothGradCAMpp(model, target_layer) as cam_extractor:
            out = model(tensor)
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5) # Resize the CAM and overlay it
        plt.figure(figsize=(6, 6))
        plt.imshow(result)
        plt.axis('off')
        plt.title(f'CAM for {model_name}')
        plt.show()


    # Confusion Matrix function
    def plot_confusion_matrix(true_labels, predictions, model_name):
        cm = confusion_matrix(true_labels, predictions)
        labels = ['Normal', 'Pneumonia']
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()

    # Train the model with baseline technique
    baseline_recall, baseline_balanced_accuracy = train_model(model, train_loader, criterion, optimizer)

    # Evaluate the model
    predictions, true_labels = evaluate_model(model, test_loader)
    f1_baseline, recall_baseline, accuracy_baseline = calculate_metrics(true_labels, predictions)
    print("Baseline Model:")
    print("F1 Score:", f1_baseline)
    print("Recall:", recall_baseline)
    print("Balanced Accuracy:", accuracy_baseline)

    #SmoothGradCAMpp for Baseline Model
    smoothgradcampp_visualization(model, test_loader, "Baseline")

    # Confusion Matrix for Baseline Model
    plot_confusion_matrix(true_labels, predictions, "Baseline")

    #Implementing Undersampling
    undersampled_majority_samples = []
    # Count samples for each class
    class_counts = {0: 0, 1: 0}
    for sample in train_data:
        class_counts[sample["label"]] += 1

    # Calculate difference in sample counts
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    difference = class_counts[majority_class] - class_counts[minority_class]

    # Separate minority and majority class samples
    minority_samples = [sample for sample in train_data if sample["label"] == minority_class]
    majority_samples = [sample for sample in train_data if sample["label"] == majority_class]

    # Undersample the majority class to match the minority class
    undersampled_majority_samples = np.random.choice(majority_samples, size=len(minority_samples), replace=False)
    # Concatenate undersampled majority samples with minority samples
    undersampled_train_data = np.concatenate([undersampled_majority_samples, minority_samples])
    # Shuffle the concatenated data
    undersampled_train_data = shuffle(undersampled_train_data)

    # Undersampling
    undersampled_train_loader = DataLoader(undersampled_train_data, batch_size=32, shuffle=True)

    undersampled_recall, undersampled_balanced_accuracy = train_model(model, undersampled_train_loader, criterion, optimizer)

    predictions, true_labels = evaluate_model(model, test_loader)
    f1_undersampled, recall_undersampled, accuracy_undersampled = calculate_metrics(true_labels, predictions)
    print("Undersampling Model:")
    print("F1 Score:", f1_undersampled)
    print("Recall:", recall_undersampled)
    print("Balanced Accuracy:", accuracy_undersampled)

    #SmoothGradCAMpp for Undersampling Model
    smoothgradcampp_visualization(model, test_loader, "Undersampling")

    # Confusion Matrix for Undersampling Model
    plot_confusion_matrix(true_labels, predictions, "Undersampling")


    #Implementing Oversampling
    oversampled_minority_samples = []
    # Oversample the minority class
    oversampled_minority_samples = np.random.choice(minority_samples, size=difference, replace=True)
    # Concatenate oversampled minority samples with majority samples
    oversampled_train_data = np.concatenate([majority_samples, oversampled_minority_samples])

    # Shuffle the concatenated data
    oversampled_train_data = shuffle(oversampled_train_data)

    # Oversampling
    oversampled_train_loader = DataLoader(oversampled_train_data, batch_size=32, shuffle=True)

    oversampled_recall, oversampled_balanced_accuracy = train_model(model, oversampled_train_loader, criterion, optimizer)

    predictions, true_labels = evaluate_model(model, test_loader)
    f1_oversampled, recall_oversampled, accuracy_oversampled = calculate_metrics(true_labels, predictions)
    print("Oversampling Model:")
    print("F1 Score:", f1_oversampled)
    print("Recall:", recall_oversampled)
    print("Balanced Accuracy:", accuracy_oversampled)

    #SmoothGradCAMpp for Oversampling Model
    smoothgradcampp_visualization(model, test_loader, "Oversampling")

    # Confusion Matrix for Oversampling Model
    plot_confusion_matrix(true_labels, predictions, "Oversampling")


    # Define data augmentation transforms for the training set
    augment_transforms = transforms.Compose([
        transforms.RandomRotation(15),  # Randomly rotate images by up to 15 degrees
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.Resize((224, 224)),  # Resize to the desired input size
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Apply transformations to train datasets
    train_dataset_aug = ChestXRayDataset(dataset["train"], transform=augment_transforms)

    # Define DataLoaders
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True)

    # Train the model
    augmentation_recall, augmentation_balanced_accuracy = train_model(model, train_loader_aug, criterion, optimizer)

    # Evaluate the model
    predictions, true_labels = evaluate_model(model, test_loader)
    f1_augmentation, recall_augmentation, accuracy_augmentation = calculate_metrics(true_labels, predictions)

    print("Augmented Model:")
    print("F1 Score:", f1_augmentation)
    print("Recall:", recall_augmentation)
    print("Balanced Accuracy:", accuracy_augmentation)

    #SmoothGradCAMpp for Augmented Model
    smoothgradcampp_visualization(model, test_loader, "Augmentation")

    # Plot confusion matrix for the augmented model
    plot_confusion_matrix(true_labels, predictions, "Augmentation")

    # Initialize BorderlineSMOTE
    borderline_smote = BorderlineSMOTE()

    # Initialize empty lists for X_train and y_train
    X_train = []
    y_train = []

    # Iterate over the DataLoader
    for batch in train_loader:
        images = batch['image'].numpy()
        labels = batch['label']
        X_train.append(images)
        y_train.append(labels)

    # Concatenate batches
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_train = np.array(X_train).reshape(-1, 3, 224, 224)  # Ensure image dimensions are correct
    y_train = np.array(y_train)

    # Move data to GPU if available
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
   
    # Reshape the flattened array to maintain image dimensions
    X_train_flat = X_train.numpy().reshape(len(X_train), -1)
    y_train_flat = y_train.numpy()

    # Apply Borderline-SMOTE to balance the training data
    X_train_borderline_smote, y_train_borderline_smote = borderline_smote.fit_resample(X_train_flat, y_train_flat)
    X_train_borderline_smote = torch.tensor(X_train_borderline_smote)
    y_train_borderline_smote = torch.tensor(y_train_borderline_smote)
    borderline_smote_train_data = TensorDataset(X_train_borderline_smote, y_train_borderline_smote)
    borderline_smote_train_loader = DataLoader(borderline_smote_train_data, batch_size=32, shuffle=True)

    # Train the model for Borderline-SMOTE
    epoch_recall_boderlineSmote = []
    epoch_balanced_accuracy__boderlineSmote = []
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for images, labels in borderline_smote_train_loader:
            optimizer.zero_grad()
            # Reshape images to match the expected input shape of the model
            images = images.view(-1, 3, 224, 224)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model after each epoch
        predictions, true_labels = evaluate_model(model, test_loader)
        _, recall, balanced_accuracy = calculate_metrics(true_labels, predictions)
        epoch_recall_boderlineSmote.append(recall)
        epoch_balanced_accuracy__boderlineSmote.append(balanced_accuracy)


    borderline_smote_recall= epoch_recall_boderlineSmote
    borderline_smote_balanced_accuracy = epoch_balanced_accuracy__boderlineSmote

    predictions, true_labels = evaluate_model(model, test_loader)
    f1_borderline_smote, recall_borderline_smote, accuracy_borderline_smote = calculate_metrics(true_labels, predictions)
    print("Borderline-SMOTE Model:")
    print("F1 Score:", f1_borderline_smote)
    print("Recall:", recall_borderline_smote)
    print("Balanced Accuracy:", accuracy_borderline_smote)

    #SmoothGradCAMpp for Borderline-SMOTE Model
    smoothgradcampp_visualization(model, test_loader, "Borderline-SMOTE")

    # Confusion Matrix for Borderline-SMOTE Model
    plot_confusion_matrix(true_labels, predictions, "Borderline-SMOTE")


    # Initialize SMOTE
    smote = SMOTE()

    # Apply SMOTE to balance the training data
    X_train_smote, y_train_smote = smote.fit_resample(X_train_flat, y_train_flat)

    # Convert back to PyTorch tensors and maintain image dimensions
    X_train_smote = torch.tensor(X_train_smote)
    y_train_smote = torch.tensor(y_train_smote)

    # Reshape back to image dimensions
    X_train_smote = X_train_smote.view(-1, 3, 224, 224)

    # Create a DataLoader with the SMOTE-augmented training data
    smote_train_data = TensorDataset(X_train_smote, y_train_smote)
    smote_train_loader = DataLoader(smote_train_data, batch_size=32, shuffle=True)

    # Train the model for SMOTE
    epoch_recall_Smote = []
    epoch_balanced_accuracy__Smote = []
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for images, labels in smote_train_loader:
            optimizer.zero_grad()
            # Reshape images to match the expected input shape of the model
            images = images.view(-1, 3, 224, 224)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model after each epoch
        predictions, true_labels = evaluate_model(model, test_loader)
        _, recall, balanced_accuracy = calculate_metrics(true_labels, predictions)
        epoch_recall_Smote.append(recall)
        epoch_balanced_accuracy__Smote.append(balanced_accuracy)

    smote_recall= epoch_recall_Smote
    smote_balanced_accuracy = epoch_balanced_accuracy__Smote

    # Evaluate the model after training
    predictions, true_labels = evaluate_model(model, test_loader)
    f1_smote, recall_smote, accuracy_smote = calculate_metrics(true_labels, predictions)

    # Print out the evaluation metrics
    print("SMOTE Model:")
    print("F1 Score:", f1_smote)
    print("Recall:", recall_smote)
    print("Balanced Accuracy:", accuracy_smote)

    #SmoothGradCAMpp for SMOTE Model
    smoothgradcampp_visualization(model, test_loader, "SMOTE")

    # Plot confusion matrix for the SMOTE model
    plot_confusion_matrix(true_labels, predictions, "SMOTE")

    # Plot recall and balanced accuracy over epochs for all techniques
    epochs = range(1, len(baseline_recall) + 1)
    plt.plot(epochs, baseline_recall, label='Baseline')
    plt.plot(epochs, undersampled_recall, label='Undersampling')
    plt.plot(epochs, oversampled_recall, label='Oversampling')
    plt.plot(epochs, borderline_smote_recall, label='Borderline SMOTE')
    plt.plot(epochs, augmentation_recall, label='Data Augmentation')
    plt.plot(epochs, smote_recall, label='SMOTE')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall over Epochs')
    plt.legend()
    plt.show()

    plt.plot(epochs, baseline_balanced_accuracy, label='Baseline')
    plt.plot(epochs, undersampled_balanced_accuracy, label='Undersampling')
    plt.plot(epochs, oversampled_balanced_accuracy, label='Oversampling')
    plt.plot(epochs, borderline_smote_balanced_accuracy, label='Borderline SMOTE')
    plt.plot(epochs, augmentation_balanced_accuracy, label='Data Augmentation')
    plt.plot(epochs, smote_balanced_accuracy, label='SMOTE')
    plt.xlabel('Epochs')
    plt.ylabel('Balanced Accuracy')
    plt.title('Balanced Accuracy over Epochs')
    plt.legend()
    plt.show()





