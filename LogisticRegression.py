#Logistic Regression model
class Multicls_log_reg(nn.Module):
    def __init__(self, n_features):
        super(Multicls_log_reg, self).__init__()
        self.linear = nn.Linear(n_features, 10)
    
    def forward(self, x):
        logits = self.linear(x)
        return logits

model_log_reg = Multicls_log_reg(n_features = 50)
criterion_log_reg = nn.CrossEntropyLoss()
optimizer_log_reg = optim.SGD(model_log_reg.parameters(), lr=0.01, momentum=0.997)
model_log_reg = train(model_log_reg, train_load_2_pca, epochs=100, lr=0.01, momentum=0.997, criterion=criterion_log_reg)

# %%
class binary_log_reg(nn.Module):
    def __init__(self, n_features):
        super(binary_log_reg, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        logits = torch.sigmoid(self.linear(x))
        return logits



# %%
def ovr_train(n , n_features , train_loader, epochs, lr, momentum, criterion):
    model = binary_log_reg(n_features)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    for e in range(epochs):
        epoch_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to('cpu'), labels.to('cpu')
            labels = (labels == n).float()  
            output = model(imgs)
            loss = criterion(output.view(-1), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        
        print(f"Epoch: {e+1}, Loss: {epoch_loss/len(train_loader)}")
    
    return model



# %%
def ovr_test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to('cpu'), labels.to('cpu')
            op = model(imgs)
            op = (op > 0.5).float()  
            all_predictions.extend(op.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (op.view(-1) == labels).sum().item()
    
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    return correct / total, conf_matrix

# %%
def train_ovr_model(train_loader, n_classes, n_features, epochs, lr, momentum):
    models = []
    for i in range(n_classes):
        print(f"Training model for class {i}")
        model = ovr_train(i, n_features, train_loader, epochs, lr, momentum, criterion=nn.BCELoss())
        models.append(model)
    return models



# %%
def test_ovr_model(models, test_loader):
    all_predictions = []
    all_labels = []
    for imgs, labels in test_loader:
        imgs, labels = imgs.to('cpu'), labels.to('cpu')
        outputs = [model(imgs) for model in models]
        outputs = torch.cat(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)  # Compute accuracy
    return accuracy, conf_matrix

n_classes = 10
n_features = 50
epochs = 100
lr = 0.01
momentum = 0.997
models = train_ovr_model(train_load_2_pca, n_classes, n_features, epochs, lr, momentum)
test_accuracy_ovr, conf_matrix_ovr = test_ovr_model(models, test_load_2_pca)
print("Confusion Matrix:")
print(conf_matrix_ovr)
print(f"Test Accuracy: {test_accuracy_ovr*100:.4f}%")
train_accuracy_ovr, _ = test(model_log_reg, train_load_2_pca)
print(f"Train Accuracy: {train_accuracy_ovr*100:.4f}%")
true_pos_ovr = np.diag(conf_matrix_ovr)
false_pos_ovr = np.sum(conf_matrix_ovr, axis=0) - true_pos_ovr
false_neg_ovr = np.sum(conf_matrix_ovr, axis=1) - true_pos_ovr
true_neg_ovr = np.sum(conf_matrix_ovr) - (true_pos_ovr + false_pos_ovr + false_neg_ovr)

precision_ovr = true_pos_ovr / (true_pos_ovr + false_pos_ovr)
recall_ovr = true_pos_ovr / (true_pos_ovr + false_neg_ovr)
f1_ovr = 2 * precision_ovr * recall_ovr / (precision_ovr + recall_ovr)
print(f"Precision_ovr: {precision_ovr}")
print(f"Recall_ovr: {recall_ovr}")
print(f"F1_ovr: {f1_ovr}")



# %%
#Plotting ROC curves
def roc_curve(y_true, y_scores):
    thresholds = np.unique(y_scores)
    thresholds = np.sort(np.append(thresholds, [1.0])) 

    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        tp = np.sum((y_scores >= threshold) & (y_true == 1))
        fn = np.sum((y_scores < threshold) & (y_true == 1))
        fp = np.sum((y_scores >= threshold) & (y_true == 0))
        tn = np.sum((y_scores < threshold) & (y_true == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return fpr_list, tpr_list

def plot_roc_curve(fpr, tpr, class_label):
    plt.plot(fpr, tpr, marker='o', label=f'Class {class_label}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_roc_curves(models, test_loader, n_classes):
    plt.figure(figsize=(8, 6))
    
    for i in range(n_classes):
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to('cpu'), labels.to('cpu')

                op = models[i](imgs).view(-1)  
                
                binary_labels = (labels == i).int()

                all_scores.extend(op.cpu().numpy())
                all_labels.extend(binary_labels.cpu().numpy())
        fpr, tpr = roc_curve(np.array(all_labels), np.array(all_scores))
        plot_roc_curve(fpr, tpr, i)

    plt.show()

def auc(fpr, tpr):
    sorted_indices = np.argsort(fpr)  # Sort FPR values
    fpr = np.array(fpr)[sorted_indices]
    tpr = np.array(tpr)[sorted_indices]
    return np.trapz(tpr, fpr)


# Computing AUC
for i in range(n_classes):
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_load_2_pca:
            imgs, labels = imgs.to('cpu'), labels.to('cpu')
            op = models[i](imgs).view(-1)

            binary_labels = (labels == i).int()
            all_scores.extend(op.cpu().numpy())
            all_labels.extend(binary_labels.cpu().numpy())

    fpr, tpr = roc_curve(np.array(all_labels), np.array(all_scores))
    auc_value = auc(fpr, tpr)
    print(f"AUC for class {i}: {auc_value:.4f}")
plot_roc_curves(models, test_load_2_pca, n_classes)
