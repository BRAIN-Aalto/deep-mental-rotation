from pathlib import Path
import torch
from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics import (
    MeanMetric,
    MinMetric,
    MaxMetric
)
import wandb


class Trainer:
    """
    """
    def __init__(
        self,
        root_dir: str,
        device: torch.device,
        logger,
        max_epochs: int = 100,
        enable_checkpointing: bool = False
    ) -> None:
        self.root_dir = Path(root_dir)
        self.logger = logger
        self.device = device
        self.max_epochs = max_epochs
        self.enable_checkpointing = enable_checkpointing

    
    def fit(
        self,
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        lr_scheduler,
        ckpt_path: str | None = None
    ):
        """
        """
        train_loss = MeanMetric().to(self.device)
        val_loss = MeanMetric().to(self.device)
        val_loss_best = MinMetric().to(self.device)

        train_acc = BinaryAccuracy().to(self.device)
        val_acc = BinaryAccuracy().to(self.device)
        val_acc_best = MaxMetric().to(self.device)

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model.load_state_dict(ckpt["model"])

            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

            start_epoch = ckpt["epoch"] + 1
            end_epoch = start_epoch + self.max_epochs
        else:
            start_epoch = 1
            end_epoch = start_epoch + self.max_epochs


        def train_epoch(epoch):
            """
            """
            model.train()

            for batch_idx, batch in enumerate(train_dataloader):
                input_1, input_2, targets = batch["image_1"].to(self.device), batch["image_2"].to(self.device), batch["label"].to(self.device)
                
                # train step
                preds = model(input_1, input_2)
                if isinstance(criterion, torch.nn.modules.loss.BCELoss):
                    loss = criterion(preds, targets)

                elif isinstance(criterion, torch.nn.modules.loss.CosineEmbeddingLoss):
                    loss = criterion(*preds, (2 * targets - 1).reshape(-1))
                    # make dummy predictions
                    preds = torch.zeros_like(targets).to(self.device)

                # clear gradients
                optimizer.zero_grad()

                # backward
                loss.backward()

                # update parameters
                optimizer.step()

                # update loss and metrics
                train_loss.update(loss)
                train_acc.update(preds, targets)

            
            train_loss_mean = train_loss.compute()
            train_acc_total = train_acc.compute()

            self.logger.info(
                f"""Epoch: {epoch}/{end_epoch}.\t[Train] Loss: {train_loss_mean:.5f}, Accuracy: {100 * train_acc_total:.2f}%."""
            )
            wandb.log({"loss/train": train_loss_mean})
            wandb.log({"acc/train": 100 * train_acc_total})

            if self.enable_checkpointing:
                ckpt_path = self.root_dir / "ckpts" / "ckpt.pth"
                ckpt_path.parent.mkdir(exist_ok=True, parents=True)
                ckpt_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss_mean,
                    "train_acc": train_acc_total
                }
                torch.save(ckpt_state, ckpt_path)


            train_loss.reset()
            train_acc.reset()


        def evaluate_epoch(epoch):
            """
            """
            model.eval()
            with torch.inference_mode():
                for batch_idx, batch in enumerate(val_dataloader):
                    input_1, input_2, targets = batch["image_1"].to(self.device), batch["image_2"].to(self.device), batch["label"].to(self.device)

                    preds = model(input_1, input_2)
                    if isinstance(criterion, torch.nn.BCELoss):
                        loss = criterion(preds, targets)

                    elif isinstance(criterion, torch.nn.CosineEmbeddingLoss):
                        loss = criterion(*preds, (2 * targets - 1).reshape(-1))
                        # make dummy predictions
                        preds = torch.zeros_like(targets).to(self.device)

                    val_loss.update(loss)
                    val_acc.update(preds, targets)

            
            val_loss_mean = val_loss.compute()
            val_loss_best.update(val_loss_mean)
            
            val_acc_total = val_acc.compute()
            val_acc_best.update(val_acc_total)

            self.logger.info(
                f"""Epoch: {epoch}/{end_epoch}.\t[Validation] Loss: {val_loss_mean:.5f}, Accuracy: {100 * val_acc_total:.2f}%, Best Accuracy: {100 * val_acc_best.compute():.2f}%.\n\n"""
            )
            wandb.log({"loss/val": val_loss_mean})
            wandb.log({"acc/val": 100 * val_acc_total})


            if val_loss_mean <= val_loss_best.compute():
                ckpt_path = self.root_dir / "ckpts" / "ckpt-best.pth"
                ckpt_path.parent.mkdir(exist_ok=True, parents=True)
                ckpt_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss_mean,
                    "val_acc": val_acc_total
                }
                torch.save(ckpt_state, ckpt_path)

                self.logger.info(f"New best model saved to {ckpt_path}.")


            val_loss.reset()
            val_acc.reset()

            

        for epoch in range(start_epoch, end_epoch):
            train_epoch(epoch)
            lr_scheduler.step()

            if val_dataloader:
                evaluate_epoch(epoch)



    def eval(
        self,
        model,
        test_dataloader,
        ckpt_path,
        criterion
    ):
        """
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model"])

        test_loss = MeanMetric().to(self.device)
        test_acc = BinaryAccuracy().to(self.device)

        test_loss.reset()
        test_acc.reset()
        
        model.eval()
        with torch.inference_mode():
            for batch_idx, batch in enumerate(test_dataloader):
                input_1, input_2, targets = batch["image_1"].to(self.device), batch["image_2"].to(self.device), batch["label"].to(self.device)

                preds = model(input_1, input_2)
                if isinstance(criterion, torch.nn.BCELoss):
                    loss = criterion(preds, targets)

                elif isinstance(criterion, torch.nn.CosineEmbeddingLoss):
                    loss = criterion(*preds, (2 * targets - 1).reshape(-1))
                    # make dummy predictions
                    preds = torch.zeros_like(targets).to(self.device)
                    
                
                test_loss.update(loss)
                test_acc.update(preds, targets)
                

        test_loss_mean = test_loss.compute()
        test_acc_total = test_acc.compute()

        return (test_loss_mean, test_acc_total)






