import torch

def train_fn(model, optimizer, criterion, train_loader, device):
    model.train()
    ctc_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        ctc_loss += loss.item()
    loss_train = ctc_loss / len(train_loader)
    return loss_train

def val_fn(model, criterion, val_loader, device):
    model.eval()
    ctc_loss = 0.0
    with torch.no_grad():
        for batch_idx, (input_encoder, input_decoder) in enumerate(val_loader):
            batch = input_encoder.shape[0]
            seq_length = input_decoder.shape[1]
            input_lengths = torch.full(size=(batch,), fill_value=seq_length, dtype=torch.long)
            target_lengths = torch.randint(low=seq_length - 2, high=seq_length, size=(batch,), dtype=torch.long)
            input_encoder, input_decoder = input_encoder.to(device), input_decoder.to(device)
            output_model = model(input_encoder, input_decoder)
            output_model = output_model.transpose(0, 1)
            loss = criterion(output_model, input_decoder, input_lengths, target_lengths)
            ctc_loss += loss.item()
        loss_val = ctc_loss / len(val_loader)
    return loss_val