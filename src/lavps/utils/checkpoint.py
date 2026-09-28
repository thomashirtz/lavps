import torch
from pathlib import Path


class CheckpointManager:
    def __init__(self, save_path: Path, enabled: bool = True, num_to_keep: int = 1, keep_list: list[int] = None):
        self.enabled = enabled
        self.save_path = save_path
        self.num_to_keep = num_to_keep
        self.keep_list = set(keep_list) if keep_list else set()
        self.save_path.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, epoch, loss):
        if not self.enabled:
            return

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }

        # Save current epoch
        current_path = self.save_path / f"epoch_{epoch:04d}.pth"
        torch.save(state, current_path)

        # Cleanup: Keep N newest OR anything in the keep_list
        all_ckpts = sorted(self.save_path.glob("epoch_*.pth"))
        if len(all_ckpts) > self.num_to_keep:
            to_remove = all_ckpts[:-self.num_to_keep]
            for fp in to_remove:
                epoch_num = int(fp.stem.split("_")[1])
                if epoch_num not in self.keep_list:
                    try:
                        fp.unlink()
                    except FileNotFoundError:
                        pass
