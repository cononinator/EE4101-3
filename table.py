import pandas as pd
from tabulate import tabulate

# Define the data in a structured dictionary
data = {
    "Model": [
        "MLP_ReLU_64_32", "MLP_LeakyReLU_64_32", "MLP_GELU_64_32", "MLP_Tanh_64_32",
        "MLP_ReLU_128_64_32", "MLP_LeakyReLU_128_64_32", "MLP_GELU_128_64_32", "MLP_Tanh_128_64_32",
        "MLP_ReLU_256_128_64_32", "MLP_LeakyReLU_256_128_64_32", "MLP_GELU_256_128_64_32", "MLP_Tanh_256_128_64_32",
        "MLP_ReLU_32_32", "MLP_LeakyReLU_32_32", "MLP_GELU_32_32", "MLP_Tanh_32_32",
        "MLP_ReLU_64", "MLP_LeakyReLU_64", "MLP_GELU_64", "MLP_Tanh_64"
    ],
    "MSE": [
        "76.2166 ± 15.6042", "86.5945 ± 11.6953", "46.4761 ± 5.1019", "80.9647 ± 16.4861",
        "36.6426 ± 4.5557", "38.0372 ± 5.4559", "36.2897 ± 4.7177", "79.6722 ± 18.2090",
        "37.5391 ± 8.2048", "32.3782 ± 2.3004", "35.2614 ± 2.1415", "128.3051 ± 80.0125",
        "102.9056 ± 9.9349", "99.9836 ± 22.1642", "56.5642 ± 8.4037", "84.7126 ± 18.0531",
        "114.9994 ± 20.6172", "115.3546 ± 14.7069", "74.9700 ± 8.0705", "57.5503 ± 9.9410"
    ],
    "RMSE": [
        "8.6870 ± 0.8672", "9.2836 ± 0.6403", "6.8074 ± 0.3680", "8.9550 ± 0.8791",
        "6.0424 ± 0.3636", "6.1521 ± 0.4344", "6.0116 ± 0.3875", "8.8703 ± 0.9950",
        "6.0931 ± 0.6432", "5.6866 ± 0.2012", "5.9354 ± 0.1791", "10.8767 ± 3.1628",
        "10.1330 ± 0.4776", "9.9295 ± 1.1784", "7.5012 ± 0.5445", "9.1536 ± 0.9616",
        "10.6783 ± 0.9868", "10.7188 ± 0.6792", "8.6460 ± 0.4653", "7.5600 ± 0.6296"
    ],
    "MAE": [
        "6.7254 ± 0.7734", "7.2991 ± 0.4526", "5.1738 ± 0.1397", "6.3609 ± 0.4615",
        "4.4878 ± 0.2383", "4.5794 ± 0.1884", "4.5252 ± 0.1965", "6.1452 ± 0.6373",
        "4.5633 ± 0.6036", "4.2167 ± 0.0939", "4.3876 ± 0.0461", "7.9254 ± 2.8829",
        "8.0914 ± 0.3240", "7.8954 ± 1.0118", "5.7385 ± 0.3803", "6.6027 ± 0.6271",
        "8.5958 ± 0.8868", "8.6077 ± 0.6220", "6.8213 ± 0.1833", "5.8045 ± 0.3349"
    ],
    "MAPE": [
        "0.2427 ± 0.0415", "0.2696 ± 0.0241", "0.1696 ± 0.0091", "0.1820 ± 0.0170",
        "0.1499 ± 0.0041", "0.1497 ± 0.0053", "0.1461 ± 0.0099", "0.1760 ± 0.0162",
        "0.1488 ± 0.0160", "0.1454 ± 0.0074", "0.1402 ± 0.0090", "0.2720 ± 0.1639",
        "0.3045 ± 0.0117", "0.2948 ± 0.0495", "0.1940 ± 0.0234", "0.1936 ± 0.0231",
        "0.3318 ± 0.0373", "0.3335 ± 0.0251", "0.2529 ± 0.0098", "0.1985 ± 0.0119"
    ],
    "R2": [
        "0.7258 ± 0.0502", "0.6875 ± 0.0414", "0.8330 ± 0.0057", "0.7116 ± 0.0273",
        "0.8682 ± 0.0090", "0.8638 ± 0.0056", "0.8698 ± 0.0058", "0.7173 ± 0.0330",
        "0.8646 ± 0.0286", "0.8828 ± 0.0111", "0.8726 ± 0.0072", "0.5455 ± 0.2759",
        "0.6294 ± 0.0182", "0.6393 ± 0.0813", "0.7967 ± 0.0207", "0.6984 ± 0.0351",
        "0.5852 ± 0.0693", "0.5830 ± 0.0542", "0.7301 ± 0.0176", "0.7941 ± 0.0177"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Print the table
print("Model Performance Results:\n")
print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
