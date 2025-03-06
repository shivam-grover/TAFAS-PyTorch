#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import math
from torch import nn
import torch.optim as optim
from utils.metrics import metric
import csv
import random

# -------------------------------
# Gated Calibration Module (GCM)
# -------------------------------

# TODO: Confirm if the torch.matmul usage is correct.

class GCM(nn.Module):
    def __init__(self, T, C, init_alpha=0.01):
        """
        Gated Calibration Module (GCM) for Test-time Adaptation in Time Series Forecasting.

        Args:
            lookback_length (int): L, the length of the look-back window.
            num_features (int): C, number of features in the time series.
        """
        super(GCM, self).__init__()
        
        # One transformation matrix W_c for each feature c, of shape (L, L)
        self.transformation_matrices = nn.ParameterList([
            nn.Parameter(torch.zeros(T, T)) for _ in range(C)
        ])
        
        # Bias terms per feature (L x 1)
        self.biases = nn.Parameter(torch.zeros(T, C))
        
        # Gating mechanism (1 per feature)
        self.gates = nn.Parameter(torch.full((C,), init_alpha))

    def forward(self, x):
        """
        Forward pass of GCM.
        
        Args:
            x (torch.Tensor): Shape (batch_size, lookback_length, num_features).
        Returns:
            torch.Tensor: Calibrated input of shape (batch_size, lookback_length, num_features).
        """
        batch_size, L, C = x.shape
        calibrated_x = torch.zeros_like(x)

        # Apply transformation per feature
        for c in range(C):
            calibrated_x[:, :, c] = torch.matmul(x[:, :, c], self.transformation_matrices[c]) + self.biases[:, c]
        
        # Apply gating mechanism
        gating_factor = torch.tanh(self.gates).unsqueeze(0).unsqueeze(0)  # Shape (1,1,C) for broadcasting
        return x + gating_factor * calibrated_x  # Final calibrated output

# -------------------------------
# PAAS: Periodicity-Aware Adaptation Scheduling
# -------------------------------
def PAAS(test_input):
    """
    Compute the partially-observed ground truth (POGT) length based on the dominant periodicity.
    Args:
        test_input (Tensor): A look-back window of shape (L, C)
    Returns:
        p (int): The determined POGT length.
    """
    # Remove mean along time axis to remove bias
    test_input = test_input - test_input.mean(dim=0)
    fft_result = torch.fft.fft(test_input, dim=0)  # shape: (L, C)
    amplitude = torch.abs(fft_result)
    # Compute power per channel and select the dominant channel
    power_per_channel = (amplitude ** 2).sum(dim=0)
    c_star = power_per_channel.argmax()
    # In the dominant channel, choose the frequency index with the highest amplitude
    freq_amplitudes = amplitude[:, c_star]
    f_star = freq_amplitudes.argmax().item()
    # Avoid division by zero; set a minimum frequency of 1
    f_star = max(f_star, 1)
    p = math.ceil(test_input.shape[0] / f_star)
    return p

# -------------------------------
# Testing Functions
# -------------------------------
def test_baseline(args, model, test_loader, device):
    """
    Evaluate the pre-trained DLinear model on the test set without adaptation.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # for timestep in range(0, 48):
            #     print(batch_x[0][336 - 48 + timestep][0].item(), batch_y[0][timestep][0].item())

            # For DLinear we assume a simple forward pass
            outputs = model(batch_x)
            f_dim = -1 if args.features == 'MS' else 0
            # print(outputs.shape,batch_y.shape)
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print(f"OG MSE: {mse}, MAE: {mae}, RSE: {rse}, Corr: {corr}")
    return mae, mse

def test_tafas(args, model, input_gcm, output_gcm, test_loader, device, adaptation_lr=1e-3):
    """
    Evaluate the pre-trained Informer/DLinear model with TAFAS test-time adaptation.

    For each adaptation cycle:
      - Compute the POGT length using PAAS on the current look-back window.
      - Collect a mini-batch of test samples (p+1 samples).
      - Compute L_partial on the first p forecast time steps using available partial ground truth.
      - Adapt only the GCM parameters based on L_partial.
      - Check if full ground truth is available for past mini-batches and compute L_full.
      - Perform prediction adjustment as per the paper.
      - Store final predictions and ground truths for evaluation.
    """

    # Freeze the source forecaster (pre-trained model)
    model.eval()

    # Set GCMs to training mode since they will be adapted
    input_gcm.train()
    output_gcm.train()

    # Optimizer for GCM parameters only
    optimizer = optim.Adam(list(input_gcm.parameters()) + list(output_gcm.parameters()), lr=adaptation_lr)
    criterion = nn.MSELoss()

    preds, trues = [], []
    test_batch = []  # Temporary buffer for forming mini-batches
    mini_batch_buffer = []  # Buffer to store past mini-batches for full GT processing
    do_PAAS = True
    period = None  # Dynamic period (p) based on PAAS

    bsz = 0  # Mini-batch counter
    current_step = 0  # Tracks the number of test samples processed
    f_dim = -1 if args.features == 'MS' else 0  # Feature selection for multi-variate settings

    # Loop through test dataset (batch_size assumed to be 1)
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        current_step += 1

        batch_x = batch_x.float().to(device)  # (1, L, C)
        batch_y = batch_y.float().to(device)  # (1, pred_len, C)

        # ----- Determine Period (p) Using PAAS -----
        if do_PAAS:
            sample = batch_x[0]  # Extract sample (L, C)
            period = PAAS(sample)  # Compute period p dynamically
            period = min(args.pred_len, period)
            cycle_size = period + 1  # Define mini-batch size
            test_batch = []  # Reset mini-batch buffer
            do_PAAS = False
            bsz = 0  # Reset mini-batch counter

        # Accumulate test samples into the mini-batch
        if bsz < cycle_size:
            test_batch.append((batch_x[0], batch_y[0]))
            bsz += 1

        if bsz < cycle_size:
            continue  # Wait until the mini-batch is full

        print("Period", period)

        # Enable PAAS for the next cycle
        do_PAAS = True

        # ----- Process Mini-Batch -----
        X_batch = torch.stack([sample[0] for sample in test_batch])  # (p+1, L, C)
        Y_batch = torch.stack([sample[1] for sample in test_batch])  # (p+1, H, C)

        # Store mini-batch with start index for later L_full computation
        mini_batch_start = current_step - period + 1
        mini_batch_buffer.append((X_batch, Y_batch, mini_batch_start))

        # ----- Compute L_full for past mini-batches if full ground truth is available -----
        mini_batches_to_remove = []
        L_full_list = []

        for (X_batch_buf, Y_batch_buf, batch_start) in mini_batch_buffer:
            if current_step >= batch_start + args.pred_len:
                # Full ground truth is available for this mini-batch
                X_cali_buf = input_gcm(X_batch_buf)  # (p, L, C)
                pred_buf = model(X_cali_buf)  # (p, H, C)
                pred_adapted_buf = output_gcm(pred_buf)  # (p, H, C)

                # Truncate predictions and ground truth to forecast length
                pred_adapted_buf = pred_adapted_buf[:, -args.pred_len:, f_dim:]
                Y_batch_buf_trunc = Y_batch_buf[:, -args.pred_len:, f_dim:]

                # Compute L_full
                L_full = criterion(pred_adapted_buf, Y_batch_buf_trunc)
                L_full_list.append(L_full)

                print(f"Mini-batch starting at {batch_start}: Full GT available. L_full = {L_full.item():.4f}, current step {current_step}")
                mini_batches_to_remove.append((X_batch_buf, Y_batch_buf, batch_start))

        # Remove processed mini-batches
        for mb in mini_batches_to_remove:
            mini_batch_buffer.remove(mb)

        # ----- Compute L_partial Using Partial Ground Truth -----
        with torch.no_grad():
            X_cali_before = input_gcm(X_batch)  # (p, L, C)
            pred_before = model(X_cali_before)  # (p, H, C)
            forecast_cali_b = output_gcm(pred_before)  # (p, H, C)

        # Extract partial ground truth (POGT) from last sample
        # POGT = Y_batch[-1][:period]
        POGT = X_batch[-1][-period : ]

        # Adaptation step: Compute L_partial
        current_sample = X_batch[0].unsqueeze(0)  # (1, L, C)
        inp_cali = input_gcm(current_sample)  # (1, L, C)
        forecast = model(inp_cali)  # (1, pred_len, C)
        forecast_cali = output_gcm(forecast)  # (1, pred_len, C)
        # print(period, forecast_cali.shape, POGT.unsqueeze(0).shape, Y_batch.shape)
        loss_p = criterion(forecast_cali[0, :period], POGT.unsqueeze(0))  # L_partial

        # Compute average L_full if any full GT mini-batches exist
        L_full_avg = sum(L_full_list) / len(L_full_list) if len(L_full_list) > 0 else 0

        # Compute total loss and update GCM parameters
        optimizer.zero_grad()
        loss = loss_p + L_full_avg
        loss.backward()
        optimizer.step()

        # ----- Prediction Adjustment (PA) -----
        with torch.no_grad():
            ipgcm = input_gcm(X_batch)  # (p, L, C)
            fc = model(ipgcm)  # (p, H, C)
            forecast_adapted = output_gcm(fc)  # (p, H, C)

            # Replace unobserved forecast values with adapted values
            for i in range(bsz - 1):
                forecast_cali_b[i, period - i:, :] = forecast_adapted[i, period - i:, :]

        # Extract final forecasted values
        forecast_cali_b = forecast_cali_b[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

        # Store predictions and ground truth for evaluation
        preds.append(forecast_cali_b[-1].detach().cpu().numpy())
        trues.append(batch_y[-1].detach().cpu().numpy())

        # Reset mini-batch buffer for next cycle
        test_batch = []

    # Convert predictions and ground truths to numpy arrays
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Compute evaluation metrics
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print(f"TAFAS MSE: {mse}, MAE: {mae}, RSE: {rse}, Corr: {corr}")

    return mae, mse

def reset_seed():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

# -------------------------------
# Main Testing Script
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test DLinear with and without TAFAS")
    # parser.add_argument('--checkpoint', type=str, required=True, help='Path to pre-trained DLinear checkpoint')
    # parser.add_argument('--data', type=str, default='ETTm1', help='Dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='Root path for data files')
    # parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='Data file name')
    # parser.add_argument('--pred_len', type=int, default=720, help='Forecasting horizon')
    # parser.add_argument('--seq_len', type=int, default=96, help='Look-back window length')
    parser.add_argument('--batch_size', type=int, default=1, help='Test batch size (preferably 1 for adaptation)')
    parser.add_argument('--use_tafas', action='store_true', help='Apply TAFAS test-time adaptation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda or cpu)')
    parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')

    args = parser.parse_args()            

    checkpoint_dirs = os.listdir("checkpoints")
    print(checkpoint_dirs)
    checkpoint_dirs.sort()

    for checkpoint in checkpoint_dirs:
        if(checkpoint == ".gitignore"):
            continue
        print("Starting grid search for ", checkpoint)
        properties = checkpoint.split("_")
        # Electricity_336_96_DLinear_custom_ftM_sl336_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0

        args.checkpoint = "checkpoints/" + checkpoint + "/checkpoint.pth"
        args.data_path = properties[0].lower() + ".csv"
        if("exchange" in args.data_path):
            args.data_path = "exchange_rate.csv"
        if("ett" in args.data_path):
            args.data_path = args.data_path.replace("ett", "ETT")
        print(args.data_path)
        args.seq_len = int(properties[1])
        args.pred_len = int(properties[2])
        args.model = properties[3]
        args.data = properties[4]
        args.features = properties[5].replace("ft", "")
        args.d_model = int(properties[9].replace("dm", ""))
        args.n_heads = int(properties[10].replace("nh", ""))
        args.e_layers = int(properties[11].replace("el", ""))
        args.d_layers = int(properties[12].replace("dl", ""))
        args.d_ff = int(properties[13].replace("df", ""))

        # Important! Note, you do not need dec_in for DLinear, but you would need it for other baselines
        
        # Important! For some reason, DLinear's univariate script has args.features as M, but args.feature as S, and args.feature is not used anywhere. Confirm before you run the pretraining.
        if(args.features == "S"):
            args.enc_in = 1
        elif(args.data_path == "electricity.csv"):
            args.enc_in = 321
        elif(args.data_path == "weather.csv"):
            args.enc_in = 21
        elif(args.data_path == "exchange_rate.csv"):
            args.enc_in = 8
        elif(args.data_path == "traffic.csv"):
            args.enc_in = 862
        elif("ETT" in args.data_path):
            args.enc_in = 7
        
        print("args", args)
        
        csv_dir = "csv"
        print(args.data_path)
        csv_file = f"{csv_dir}/{args.data_path.replace('.csv', '')}_{args.seq_len}_p{args.pred_len}_{args.features}.csv"
        print(csv_file)
        with open(csv_file, 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(["lr", "alpha", "val_mse", "val_mae", "test_mse", "test_mae", "og_mse", "og_mae", "imp mse", "imp mae"])
            
        # Get OG MSE and MAE
        # I will do this later because right now I am running og tests with batch size 1
    
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
        # Load test data using the provided data provider (assumed to be similar to training script)
        from data_provider.data_factory import data_provider

        search_space = {
            "adaptation_lr": [0.005, 0.003, 0.001, 0.0005, 0.0001],
            "alpha": [0.01, 0.05, 0.1, 0.3]
        }

        # For grid search, run TAFAS on both validation and test loss. We optimise for validation loss, and test losses are just for reference.
        from models import DLinear
        args.batch_size = 1

        for alr in search_space['adaptation_lr']:
            for alpha in search_space['alpha']:

                print("Testing with lr", alr, "alpha", alpha)

                _, test_loader = data_provider(args, flag='test')
                _, vali_loader = data_provider(args, flag='val')

                # print(vali_loader)

                # Load the pre-trained DLinear model
                model = DLinear.Model(args).float().to(device)
                model.load_state_dict(torch.load(args.checkpoint, map_location=device))
                model.eval()
        
                # Initialize the two GCM modules:
                # Input GCM uses the look-back length (seq_len) and output GCM uses the forecast horizon (pred_len)
                num_channels = args.enc_in  # For DLinear, enc_in equals the number of channels (e.g., 7)
                input_gcm = GCM(T=args.seq_len, C=num_channels, init_alpha=alpha).to(device)
                output_gcm = GCM(T=args.pred_len, C=num_channels, init_alpha=alpha).to(device)
                
                reset_seed()
                print("Validation with TAFAS adaptation...")
                val_mae, val_mse = test_tafas(args, model, input_gcm, output_gcm, vali_loader, device, adaptation_lr=alr)    

                # Reinitialise GCM for test set. This is only for reference.
                num_channels = args.enc_in  # For DLinear, enc_in equals the number of channels (e.g., 7)
                input_gcm = GCM(T=args.seq_len, C=num_channels, init_alpha=alpha).to(device)
                output_gcm = GCM(T=args.pred_len, C=num_channels, init_alpha=alpha).to(device)

                reset_seed()
                print("Testing with TAFAS adaptation...")
                test_mae, test_mse = test_tafas(args, model, input_gcm, output_gcm, test_loader, device, adaptation_lr=alr)

                with open(csv_file, 'a', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow([alr, alpha, val_mse, val_mae, test_mse, test_mae])

        # # Save the predictions and ground truths to disk
        # os.makedirs('./results', exist_ok=True)
        # np.save('./results/predictions.npy', preds)
        # np.save('./results/ground_truth.npy', trues)
        # print("Testing completed. Results saved to './results'.")

if __name__ == '__main__':
    main()