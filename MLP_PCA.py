#MLP_PCA model
import numpy as np
import matplotlib.pyplot as plt

class my_PCA():
    def __init__(self, n_comp):
        self.n_comp = n_comp
        self.mu = None
        self.std = None
        self.components = None
    
    def fit_transform(self, X):
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-10 
        X_norm = (X - self.mu) / self.std
        
        cov_matrix = np.cov(X_norm, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        dec_ord = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, dec_ord]
        
        self.components = eigenvectors[:, :self.n_comp]
        
        return X_norm @ self.components

    def transform(self, X):
        X_norm = (X - self.mu) / self.std
        return X_norm @ self.components
    
    def inverse_transform(self, X_pca):
        return X_pca @ self.components.T * self.std + self.mu

images_train2 = []
labels_train2 = []
for img, label in train_data_2:
    images_train2.append(img.flatten().numpy())
    labels_train2.append(label)

images_train2 = np.array(images_train2)
labels_train2 = np.array(labels_train2)

images_test2 = []
labels_test2 = []
for img, label in test_data_2:
    images_test2.append(img.flatten().numpy())
    labels_test2.append(label)

images_test2 = np.array(images_test2)
labels_test2 = np.array(labels_test2)

pca = my_PCA(n_comp=50)
train_data_2_pca = pca.fit_transform(images_train2)
test_data_2_pca = pca.transform(images_test2)

img_for_recon = images_train2[4801].reshape(28, 28)

pca_values = [50, 250, 500, 784]
def plot_reconstructions(pca_values):
    for val in pca_values:
        pca = my_PCA(n_comp=val)
        train_data_2_pca = pca.fit_transform(images_train2)
        train_data_2_recon = pca.inverse_transform(train_data_2_pca)
        img_recon = train_data_2_recon[4801].reshape(28, 28)
    
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_for_recon, cmap='gray')
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        
        ax[1].imshow(img_recon, cmap='gray')
        ax[1].set_title(f"Image after PCA with {val} components")
        ax[1].axis('off')
    
    plt.show()
 
plot_reconstructions(pca_values)   

print("Original shape: ", images_train2.shape)
print("Reduced shape: ", train_data_2_pca.shape)

# %%


class MLP_PCA(nn.Module):
    def __init__(self):
        super(MLP_PCA, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
model_mlp_pca = MLP_PCA()
criterion_mlp_pca = nn.CrossEntropyLoss()

train_data_2_pca = torch.tensor(train_data_2_pca, dtype=torch.float32)
test_data_2_pca = torch.tensor(test_data_2_pca, dtype=torch.float32)

train_load_2_pca = DataLoader(list(zip(train_data_2_pca, labels_train2)), batch_size=64, shuffle=True)
test_load_2_pca = DataLoader(list(zip(test_data_2_pca, labels_test2)), batch_size=64, shuffle=False)

model_mlp_pca = train(model_mlp_pca, train_load_2_pca, epochs=100, lr=0.01, momentum=0.99, criterion=criterion_mlp_pca)

test_accuracy_mlp_pca, conf_matrix_mlp_pca = test(model_mlp_pca, test_load_2_pca)

print(f"Test Accuracy: {test_accuracy_mlp_pca*100:.4f}%")
print("Confusion Matrix:")
print(conf_matrix_mlp_pca)

train_accuracy_mlp_pca, _ = test(model_mlp_pca, train_load_2_pca)
print(f"Train Accuracy: {train_accuracy_mlp_pca*100:.4f}%")

true_pos_mlp_pca = np.diag(conf_matrix_mlp_pca)
false_pos_mlp_pca = np.sum(conf_matrix_mlp_pca, axis=0) - true_pos_mlp_pca
false_neg_mlp_pca = np.sum(conf_matrix_mlp_pca, axis=1) - true_pos_mlp_pca
true_neg_mlp_pca = np.sum(conf_matrix_mlp_pca) - (true_pos_mlp_pca + false_pos_mlp_pca + false_neg_mlp_pca)

precision_mlp_pca = true_pos_mlp_pca / (true_pos_mlp_pca + false_pos_mlp_pca)
recall_mlp_pca = true_pos_mlp_pca / (true_pos_mlp_pca + false_neg_mlp_pca)
f1_mlp_pca = 2 * precision_mlp_pca * recall_mlp_pca / (precision_mlp_pca + recall_mlp_pca)

print(f"Precision_mlp_pca: {precision_mlp_pca}")
print(f"Recall_mlp_pca: {recall_mlp_pca}")
print(f"F1_mlp_pca: {f1_mlp_pca}")
