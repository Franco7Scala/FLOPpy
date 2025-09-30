
device = "cpu" # cuda

cnn_learning_rate = 1e-3
cnn_momentum = 0.5


model = Cnn(device)
optimizer = optim.SGD(model.parameters(), lr=cnn_learning_rate, momentum=cnn_momentum)
criterion = None
