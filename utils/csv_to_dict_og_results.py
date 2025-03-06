import csv
import io

csv_data = """Model, data, features, pred_len, mse, mae, rse
DLinear,exchange_rate,M,96,0.08602219,0.21504414,0.22338408
DLinear,exchange_rate,M,192,0.16539237,0.30525056,0.31452128
DLinear,exchange_rate,M,336,0.37254515,0.46139312,0.47553933
DLinear,exchange_rate,M,720,1.0981272,0.78847796,0.8264957
DLinear,electricity,M,96,0.14028244,0.23781581,0.3725387
DLinear,electricity,M,192,0.15328997,0.25073147,0.38944605
DLinear,electricity,M,336,0.16738252,0.2672931,0.40782088
DLinear,electricity,M,720,0.20223585,0.300549,0.4490328
DLinear,ETTh1,M,96,0.3818876,0.40546358,0.5866297
DLinear,ETTh1,M,192,0.414329,0.42476973,0.61125773
DLinear,ETTh1,M,336,0.43626896,0.4428016,0.6294406
DLinear,ETTh1,M,720,0.46786636,0.48996738,0.65578103
DLinear,ETTh2,M,96,0.35995933,0.41160414,0.48205957
DLinear,ETTh2,M,192,0.377632,0.41109437,0.49276054
DLinear,ETTh2,M,336,0.45047522,0.4642693,0.5361585
DLinear,ETTh2,M,720,0.70776296,0.59854573,0.67307496
DLinear,traffic,M,192,0.41927686,0.2857202,0.532655
DLinear,traffic,M,96,0.41029942,0.28178176,0.52972674
DLinear,traffic,M,336,0.4323104,0.29399732,0.53944564
DLinear,traffic,M,720,0.4651648,0.31485903,0.55724764
DLinear,ETTm1,M,96,0.17057961,0.26780722,0.33458713
DLinear,ETTm1,M,192,0.23227312,0.31377488,0.39010715
DLinear,ETTm1,M,336,0.31618354,0.37609,0.45366335
DLinear,ETTm1,M,720,0.45253125,0.4602141,0.54010016
"""

# Initialize an empty dictionary
data_dict = {}

# Use StringIO to simulate a file object from the string
f = io.StringIO(csv_data)
# skipinitialspace=True removes extra spaces after commas
reader = csv.DictReader(f, skipinitialspace=True)

for row in reader:
    # Build a key of the form "data_features_pred_len"
    key = f"{row['data']}_{row['features']}_{row['pred_len']}"
    data_dict[key] = {
        "mse": float(row["mse"]),
        "mae": float(row["mae"]),
        "rse": float(row["rse"])
    }

print(data_dict)