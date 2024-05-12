
# PatchTST Compression Techniques

## Description

This project explores the application of compression techniques on PatchTST, a model designed for multivariate time series forecasting and self-supervised representation learning. PatchTST introduces patching, which segments time series into subseries-level patches, and channel-independence, where each channel contains a single univariate time series. The core architecture utilizes a vanilla Transformer encoder. The model compression techniques we have applied on PatchTST are LoRA and pruning.

## Project Milestones

### Milestones in project proposal:

1. **Model Setup on HPC:** Completed
2. **Implementation of PatchTST:** Completed
3. **Profiling PatchTST:** Completed
4. **Implementing LoRA on PatchTST:** Completed
5. **Pruning PatchTST:** Completed

### Some extra things we tried out:

- **Post Training Quantization on PatchTST:** In progress
- **Quantization Aware Training (QAT) on PatchTST:** In progress

## Repository Structure

The repository contains the following files:

- `main.py`: Main script to run training, fine-tuning, and testing of PatchTST.
- `model.py`: Implementation of the PatchTST model.
- `data.py`: Generates the datasets for training, fine-tuning, validation, and testing.
- `datasets.py`: Data preprocessing and loading functions.
- `saved_models/`: Directory to save trained model checkpoints.
- `results/`: Directory to store experimental results.

## Example Commands

### Training

To train PatchTST from scratch on the ETTh1 dataset:

```
python main.py
```

To fine-tune PatchTST:

```
python main.py --mode=finetune --ckpt-load=path/to/ckpt
```

To fine-tune PatchTST with LoRA:

```
python main.py --mode=finetune --lora=True --ckpt-load=path/to/ckpt
```

For ETTh dataset, the number of channels passed should be 7, and dataset name in `data.py` should be changed to `etth1.csv`. For weather dataset, input channels should be 21, and dataset name should be changed to `weather.csv`.

## Results and Observations

### LoRA

The following results are after running the model for 10 epochs, on the ETTh dataset:

- **Fine-tuning without LoRA:**
  - Total training time: 260.36s
  - Test MSE: 6.12
  - Trainable Params: 810,094 (99.99% of total)
  - Model Size: 3.27 MB

- **Fine-tuning with LoRA:**
  - Total training time: 9.53s
  - Test MSE: 5.80
  - Trainable Params: 49,152 (5.72% of total)
  - Model Size: 3.47 MB

### Pruning

Future Directions

Trying compression techniques on different bigger time series model.

## References

1. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. arXiv preprint arXiv:2211.14730. Retrieved from [https://arxiv.org/abs/2211.14730](https://arxiv.org/abs/2211.14730)
2. PyTorch Quantization Documentation: [https://pytorch.org/docs/stable/quantization.html](https://pytorch.org/docs/stable/quantization.html)
3. PatchTST Repository: [https://github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)
