import os
import argparse
import logging
import torch
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import pandas as pd
import loralib as lora
from time import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.utils.prune as prune

from data2 import CustomDataset
from model import PatchTST

class Learner:
    def __init__(self, device):
        
        self.device = device
    
    def load_data(self, trainmode, data_path, shuffle):

        data = CustomDataset(data_path=data_path, trainmode=trainmode)
        logger.info(f"Data loaded from {data_path}. Num samples: {len(data)}.")
        return torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=shuffle)

    def adjust_learning_rate(self, steps, optimizer, warmup_step=300):
        if steps**(-0.5) < steps * (warmup_step**-1.5):
            lr_adjust = (16**-0.5) * (steps**-0.5) * args.adjust_factor
        else:
            lr_adjust = (16**-0.5) * (steps * (warmup_step**-1.5)) * args.adjust_factor

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_adjust

        return optimizer

    def test(self, model, dataloader=None):
        model = model.to(self.device)
        if dataloader is None:
            dataloader = self.load_data(False, args.test_path, False)

        model.eval()
        criterion = torch.nn.MSELoss()

        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                preds = model(inputs)
                preds = preds[:, -args.target_window:, -1:].squeeze(-1)
                loss = criterion(preds, targets)
                total_loss += loss.item()

        total_loss /= len(dataloader)

        return total_loss

    def train(self, model):
        model = model.to(self.device)
        if args.lora:
            lora.mark_only_lora_as_trainable(model)

        trainloader = self.load_data(True, args.train_path, True)
        valloader = self.load_data(True, args.val_path, False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()

        best_val_loss = np.inf
        train_history = []
        valid_history = []
        train_steps = 1

        if args.adjust_lr:
            optimizer = self.adjust_learning_rate(train_steps, optimizer)

        logger.info("Starting training")
        # module = model.backbone.encoder.layers[0].self_attn.to_out[0]
        # prune.random_unstructured(module, name="weight", amount=0.5)
        start = time()
        for epoch in range(args.epochs):
            model.train()

            if ((epoch+1)>20) and ((epoch+1)%5==0):  
                for name, module in model.named_modules():
                    # prune 10% of connections in all linear layers
                    if isinstance(module, torch.nn.Linear):
                        logger.info(f"Pruning module : {name}")
                        module = model.backbone.encoder.layers[0].self_attn.to_out[0]
                        prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=0)
                        # prune.random_unstructured(module, name="weight", amount=0.2)
                        prune.remove(module, 'weight')

            epoch_loss = 0
            for inputs, targets in trainloader:
                optimizer.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                pred_y = model(inputs)
                pred_y = pred_y[:, -args.target_window:, -1:].squeeze(-1)
                loss = criterion(pred_y, targets)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_steps += 1

            if args.adjust_lr:
                optimizer = self.adjust_learning_rate(train_steps, optimizer)

            epoch_loss /= len(trainloader)
            val_loss = self.test(model=model, dataloader=valloader)
            logger.info(f"epoch: {epoch} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

            if best_val_loss >= val_loss:

                best_val_loss = val_loss

                checkpoint = {
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{args.ckpt_dir}/{args.mode}-{(train_steps):05d}.pth"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            train_history.append(epoch_loss)
            valid_history.append(val_loss)

        end = time() 
        total_time = end-start
        logger.info(f"Training completed. Time taken: {total_time / 60:.3f} mins")

        if args.profile == True:
            # Set up the directory for saving the results 
            results_dir = os.path.join('profiling_results', datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
            os.makedirs(results_dir)
            


            data = { 'train_loss':train_history, 'valid_loss':valid_history }
            loss_df = pd.DataFrame(data)
            loss_df.to_csv(os.path.join(results_dir, 'losses.csv'), index=False) 

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(np.arange(1, args.epochs+1), valid_history, label="validation", linestyle = '--', color = 'mediumvioletred')
            ax.plot(np.arange(1, args.epochs+1), train_history, label="training", color = 'darkslateblue' )
            ax.set_xlabel("epochs")
            ax.set_ylabel("MSE Error")
            ax.set_ylim(0,1)
            ax.legend()
            ax.figure.savefig(os.path.join(results_dir, 'loss_curves.png'))

            input = next(iter(valloader))[0].to('cuda')

            with profile(activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    model.eval()
                    model(input)

            with open(os.path.join(results_dir, 'profiler.txt'), "w") as f:
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            prof.export_chrome_trace(os.path.join(results_dir, 'trace.json'))

            trainable, total = get_trainable_parameters(model)
            percent = (trainable/total) *100

            with open(os.path.join(results_dir, 'metrics.txt'), "w") as f:
                f.write(f"Total training time: {total_time}s \nTrainable Params: {trainable}, {percent}% of total \nModel Size: {print_model_size(model)} MB")
        return model
def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    size = os.path.getsize("tmp.pt")/1e6
    os.remove('tmp.pt')
    return size

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/{args.mode}-log.txt")]
    )
    logger = logging.getLogger(__name__)
    
    return logger

def get_trainable_parameters(model):
    trainable_params = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return trainable_params, total_params

def print_model_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    size = os.path.getsize("tmp.pt")/1e6
    # print("Size of model: %.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')
    return size

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", help="Options: train, finetune, test")
    parser.add_argument("--log-dir", type=str, default="log", help="directory for the log file")
    parser.add_argument("--ckpt-dir", type=str, default="saved_models", help="directory to save model ckpt")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="dataset", help="Path to data files")
    parser.add_argument("--epochs", type=int, default=5, help="Num epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch Size")
    parser.add_argument("--lora", type=bool, default=False, help="Use LoRA to train")

    parser.add_argument("--adjust-lr", type=bool, default=True, help="To adjust learning rate")
    parser.add_argument("--adjust-factor", type=float, default=1e-3, help="Adjust factor if adjust-lr is true")
    parser.add_argument("--c-in", type=int, default=7, help="Num in channels")
    parser.add_argument("--context-window", type=int, default=336, help="Context window size")
    parser.add_argument("--target-window", type=int, default=96, help="Target window size")
    parser.add_argument("--patch-len", type=int, default=16, help="Path length")
    parser.add_argument("--stride", type=int, default=8, help="Stride")
    parser.add_argument("--profile", type=bool, default=True, help="To produce profiling results")

    args = parser.parse_args()

    assert os.path.exists(args.data_path), f"{args.data_path} does not exist"

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = create_logger(args.log_dir)

    # Choose GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    learner = Learner(device)

    model = PatchTST(c_in=args.c_in, context_window=args.context_window, target_window=args.target_window, 
                     patch_len=args.patch_len, stride=args.stride, use_lora=args.lora)
    model = model.to(device)
    if args.mode == "train":
        args.train_path = f"{args.data_path}/train.csv"
        args.val_path = f"{args.data_path}/val.csv"

        model = learner.train(model)
        
        trainable_params, total_params = get_trainable_parameters(model)
        logger.info(f"% Trainable Parameters: {(trainable_params / total_params) * 100: .3f}%")
        logger.info(f" Size of Model: {print_model_size(model)}")

    elif args.mode == "finetune":
        args.train_path = f"{args.data_path}/finetune.csv"
        args.val_path = f"{args.data_path}/val.csv"

        if args.lora:
            args.mode = f"{args.mode}-lora"

        assert args.ckpt is not None, "No pre-trained ckpt provided for fine-tuning"
        assert os.path.exists(args.ckpt), f"{args.ckpt} does not exist"

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"], strict=False)
        model = model.to(device)
        logger.info(f" Size of Model before pruning: {print_model_size(model)} MB")
        model = learner.train(model)

        trainable_params, total_params = get_trainable_parameters(model)
        logger.info(f"% Trainable Parameters: {(trainable_params / total_params) * 100: .3f}%")
        logger.info(f" Size of Model after pruning: {print_model_size(model)} MB")

    elif args.mode == "test":

        args.test_path = f"{args.data_path}/test.csv"

        assert args.ckpt is not None, "No ckpt provided for testing"
        assert os.path.exists(args.ckpt), f"{args.ckpt} does not exist"

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"], strict=False)
        model = model.to(device)

        loss = learner.test(model)
        logger.info(f"Test Loss: {loss:.3f}")
    
    else:
        raise NotImplementedError(f"{args.mode} is not a valid mode")
