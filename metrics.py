class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_metric = float('inf')  # Initialize with positive infinity for loss
        self.early_stop = False

    def __call__(self, current_metric):
        if self.best_metric - current_metric > self.delta:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
        return self.early_stop

def accuracy_fn(y_true, y_pred):
    correct = sum([i == j for i, j in zip(y_true, y_pred)])
    acc = (correct / len(y_pred))
    return acc