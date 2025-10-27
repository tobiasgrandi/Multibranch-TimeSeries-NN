# Multibranch-TimeSeries-NN
## Project Description
This project explores the behavior of a multi-branch Recurrent Neural Network (RNN) for predicting YouTube video views. The network consists of two parallel branches: one using LSTM and the other using GRU, which are then combined through a final linear layer.  

For comparison, two additional single-branch models were implemented: one with a single LSTM layer and another with a single GRU layer. The goal is to visualize and analyze how different RNN architectures perform on time series forecasting tasks.  

The project is implemented using **PyTorch** and leverages a YouTube dataset containing video metrics.

## Project Structure
```
├── .venv
├── .vscode
├── src
│ ├─ 📂 data                    # Data preprocessing and split
│ ├─ 📂 helpers       
│ │ ├── EarlyStopping.py        # EarlyStopping implementation
│ │ ├── Trainer.py              # Trainer class
│ │ ├── Tester.py               # Tester class
│ │ └── YoutubeDataset.py       # PyTorch Dataset class
│ ├─ 📂 model
│ │ ├── GRULayer.py             # Personalized GRU layer
│ │ ├── GRUModel.py             # GRU network
│ │ ├── LSTMLayer.py            # Personalized LSTM layer
│ │ ├── LSTMModel.py            # LSTM network
│ │ └── MultiBranchModel.py     # MultiBranch network
│ └─ 📂 models                  # Trained networks .pt files
│ └─ 📂 results                 # Training and evaluation results
├── training.ipynb # Notebook to train and evaluate models
```

## Installation

This project uses **UV** and requires Python >= 3.10. You can install the dependencies with:

```
uv sync
```

## Usage
- Preprocess Data
    -

    All preprocessing steps, including train/validation/test split, are implemented in `src/data/data_work.ipynb`.

- Train Models
    - 
    You can train the three models (LSTM, GRU, and Multi-Branch) using the training.ipynb notebook. Key parameters include:
    
    * `seq_len`: lenght of input sequences. Number of past steps used to forecast.
    * `forecast_horizon`: Number of steps to predict.
    * `learning_rate`, `epochs`, `batch_size`, etc.

- Evaluate Models
    -
    Evaluation scripts and helper classes are in `src/helpers/`. Metrics and predictions are saved in `src/results/`. The notebook `training.ipynb` also generates comparison plots (`loss.png`, `metrics.png`).

## Data
The project uses a YouTube dataset containing the following columns:
* `videostatsid`: Row id.
* `ytvideoid`: The id of the video according to Youtube.
* `views`
* `comments`
* `likes`
* `dislikes`

Original dataset can be found here [Youtube dataset](https://github.com/jettisonthenet/timeseries_trending_youtube_videos_2019-04-15_to_2020-04-15).

The dataset is preprocessed and split into `train_data.csv`, `val_data.csv`, and `test_data.csv` for model training and evaluation.

## Results
The project produces:
* `loss.png`: Loss curves for each model.
* `metrics.png`: Comparative metrics (Loss, MSE, MAE, RMSE, R2).
* `results.csv`: Tabular comparison of model predictions vs actual values.

![Loss](/src/results/loss.png)
![Metrics](/src/results/metrics.png)

> **Observation**
>
> While the performance of all three models (LSTM, GRU, and Multi-Branch) is quite similar in terms of prediction evaluation metrics, the Multi-Branch model tends to converge in fewer epochs compared to the single-branch models. This suggests that combining both LSTM and GRU layers allows the network to learn more efficiently from the sequence data.
