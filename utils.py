import os

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"[CHECKPOINT] Sauvegardé à {path}")
    

def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        print(f"[CHECKPOINT] Aucun checkpoint trouvé à {path}")
        return 0  # epoch de départ
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"[CHECKPOINT] Chargé depuis {path}, reprise à l'époque {start_epoch}")
    return start_epoch