# TAFAS Unofficial Implementation

This is a Pytorch implementation of TAFAS: "[Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation](https://arxiv.org/pdf/2501.04970)". 


## TODO
- [x] GCM
- [x] PAAS
- [x] Full loss when previous ground truth is available
- [x] Prediction Adjustment
- [x] Grid search for alpha and adaptation learning rate
- [x] Implement with DLinear
- [ ] Implement with other baselines
- [ ] Reproduce all experiments

## Steps

### Setup your environment
First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n DLinear python=3.6.9
conda activate DLinear
pip install -r requirements.txt
```

### Data Preparation

You can obtain all the nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

### Pretrain a model
Navigate to the directory `scripts/EXP-LongForecasting/DLinear` and run a script of your choice to obtain a pretrained model.
**NOTE**: The TAFAS paper uses a batch size of 64, and 30 epochs. So you would need to change that everywhere.

### Run TAFAS on the pretrained model

As an example, the following command runs TAFAS on a pretrained checkpoint for DLinear with ETTh1 and a prediction length of 720. Make sure to use the same values for parameters that you used while pretraining.
**Note**: According to me, batch size during testing should be 1. I will contact the authors to ensure if I got this right, but empirically also, batch size 1 is the best as well.

```
python3.8 -u run_TAFAS.py \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 1\
  --checkpoint "/home/23bx19/TTA/TAFAS/DLinear/checkpoints/ETTh1_336_720_DLinear_ETTh1_ftM_sl336_ll48_pl720_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"\
  --use_tafas
  ```


  ### Grid search if necessary

  The paper mentions that they optimise the hyperparameters alpha and adaptation learning rate. For this you can run the following command.

  ```
python3.8 -u run_TAFAS_grid_search.py \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 1\
  --checkpoint "/home/23bx19/TTA/TAFAS/DLinear/checkpoints/ETTh1_336_720_DLinear_ETTh1_ftM_sl336_ll48_pl720_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"\
  ```

  This saves a csv file in the "csv" directory where you can observe the grid search results in order to pick the optimal hyperparameters.


This work is based on the paper "[Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation](https://arxiv.org/pdf/2501.04970)" and "[Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/pdf/2205.13504.pdf)".