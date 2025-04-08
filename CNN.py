#CNN model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_stack = nn.Sequential(
            nn.Conv2d(64, 10, kernel_size=1), 
        )
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.global_avg_pool(x) 
        x = self.linear_stack(x) 
        return x.view(x.size(0), -1) 

model_cnn = CNN()
criterion_cnn = nn.CrossEntropyLoss()

model_cnn = train(model_cnn, train_load_2, epochs=100, lr=0.01, momentum=0.999, criterion=criterion_cnn)

test_accuracy_cnn, conf_matrix_cnn = test(model_cnn, test_load_2)

print(f"Test Accuracy: {test_accuracy_cnn*100:.4f}%")
print("Confusion Matrix:")
print(conf_matrix_cnn)

train_accuracy_cnn, _ = test(model_cnn, train_load_2)
print(f"Train Accuracy: {train_accuracy_cnn*100:.4f}%")

true_pos_cnn = np.diag(conf_matrix_cnn)
false_pos_cnn = np.sum(conf_matrix_cnn, axis=0) - true_pos_cnn
false_neg_cnn = np.sum(conf_matrix_cnn, axis=1) - true_pos_cnn
true_neg_cnn = np.sum(conf_matrix_cnn) - (true_pos_cnn + false_pos_cnn + false_neg_cnn)

precision_cnn = true_pos_cnn / (true_pos_cnn + false_pos_cnn)
recall_cnn = true_pos_cnn / (true_pos_cnn + false_neg_cnn)
f1_cnn = 2 * precision_cnn * recall_cnn / (precision_cnn + recall_cnn)

print(f"Precision_cnn: {precision_cnn}")
print(f"Recall_cnn: {recall_cnn}")
print(f"F1_cnn: {f1_cnn}")
