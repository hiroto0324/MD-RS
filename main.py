import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import time
import re
from scipy.stats import gamma
import argparse


def generate_Wr(Nx, alpha_r, beta_r):
    np.random.seed(0)
    # np.random.seed(123456)
    Wr = np.zeros(Nx * Nx)
    Wr[0:int(Nx * Nx * beta_r)] = np.random.normal(0, 1 / np.sqrt(beta_r * Nx), int(Nx * Nx * beta_r))
    np.random.shuffle(Wr)
    Wr = Wr.reshape((Nx, Nx))
    Wr *= alpha_r
    return Wr

def generate_Wb(Nx, Ny, alpha_b, beta_b):
    np.random.seed(0)
    # np.random.seed(234567)
    Wb = np.zeros(Nx * Ny)
    Wb[0:int(Nx * Ny * beta_b)] = np.random.uniform(-1, 1, int(Nx * Ny * beta_b)) # beta_b = 非ゼロ率
    np.random.shuffle(Wb)
    Wb = Wb.reshape((Nx, Ny))
    Wb *= alpha_b
    return Wb

def generate_Wi(Nx, Ni, alpha_i, beta_i):
    np.random.seed(0)
    # np.random.seed(345678)
    Wi = np.zeros(Nx * Ni)
    Wi[0:int(Nx * Ni * beta_i)] = np.random.uniform(-1, 1, int(Nx * Ni * beta_i)) # beta_b = 非ゼロ率
    np.random.shuffle(Wi)
    Wi = Wi.reshape((Nx, Ni))
    Wi *= alpha_i
    return Wi

def generate_Wi_separated(Nx, Ni, alpha_i, beta_i):
    np.random.seed(0)
    # np.random.seed(456789)
    Wi = np.zeros((Nx, Ni))
    Wi_ = np.zeros(int(Nx / Ni))
    for i in range(Ni):
        Wi_[:int(beta_i * Nx / Ni)] = np.random.uniform(-1, 1, int(beta_i * Nx / Ni)) # beta_b =
        np.random.shuffle(Wi_)
        Wi[i * int(Nx / Ni): (i + 1) * int(Nx / Ni), i] = Wi_ # i番目の入力をうけるニューロン達
    np.random.shuffle(Wi) # 行方向にシャッフル(受け入れニューロンをシャッフル)
    Wi *= alpha_i
    return Wi

def fx(x):
    return np.tanh(x)
    # return x * (x > 0) # ReLU
    # return (1 + np.tanh((1/2) * x)) / 2

def dfx(x):
    return 1.0 / (np.cosh(x) * np.cosh(x))

def fx_modulated(x):
    return amp_act * np.tanh(gain_act * x - thr_act)

def f_kernel(x):
    return x
    # return np.tanh(x)

# 出力層の活性化関数
def fy(x):
    # return np.tanh(b * x)
    # return (1 + np.tanh((b / 2) * x)) / 2
    return x


def fyi(x):
    # return np.arctanh(x)
    # return (2/b) * np.arctanh(2 * x - 1)
    return x


# エラー層の活性化関数
def fr(x):
    # return np.fmax(0, x)
    return x


def ridge_regression(X, Y, alpha):
    W = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ Y
    return W

class Reservoir:
    def __init__(self, param):
        self.T1 = param['T1']
        self.train_length = param['train_length']
        self.Nx = param['Nx']
        self.Ny = param['Ny']
        self.Ni = param['Ni']
        self.N_P = param['N_P']

        self.tau = param['tau']
        self.dt = param['dt']
        self.sigma_noise = param['sigma_noise']
        self.alpha = param['alpha']

        self.alpha_r = param['alpha_r']
        self.beta_r = param['beta_r']
        self.alpha_i = param['alpha_i']
        self.beta_i = param['beta_i']
        self.alpha_b = param['alpha_b']
        self.beta_b = param['beta_b']

        ## 精度行列Pをラベルの数だけ用意
        self.P = np.identity(self.N_P) / self.alpha
        self.P_mod = np.zeros_like(self.P)
        self.mu = np.zeros(self.N_P)


    def generate_weight_matrices(self):
        self.Wr = generate_Wr(self.Nx, self.alpha_r, self.beta_r)
        self.Wb = generate_Wb(self.Nx, self.Ny, self.alpha_b, self.beta_b)
        self.Wi = generate_Wi(self.Nx, self.Ni, self.alpha_i, self.beta_i)
        # self.Wi = generate_Wi_separated(self.Nx, self.Ni, self.alpha_i, self.beta_i)
        self.Wo = np.zeros((self.Ny, self.N_P))

    def load_weight_matrices(self, Wr, Wb, Wi):
        self.Wr = Wr * alpha_r
        self.Wb = Wb * alpha_b
        self.Wi = Wi * alpha_i
        self.Wo = np.zeros((self.Ny, self.N_P))
        self.Wi[Wi_zero_idx, 0] = 0
        self.Wb[Wb_zero_idx, 0] = 0

    ## 学習又は推論の開始時にリザバーを初期値にをリセット
    def reset(self, x_init=0.2):
        self.X = np.zeros((self.T1, self.Nx))
        self.Y = np.zeros((self.T1, self.Ny))
        # self.X[0, :] = np.random.uniform(-1, 1, self.Nx) * 0.2 # initial state
        self.X[0, :] = x_init
        self.Y[0, :] = np.zeros(self.Ny)
        # self.TRAKR_errors = np.zeros((self.T1, self.Ny))
        self.TRAKR_errors = np.zeros(self.T1)

    def freerun(self):
        self.reset()
        self.T_free = 10000
        self.X_ = np.zeros((self.T_free, self.Nx))
        self.Y_ = np.zeros((self.T_free, self.Ny))
        self.X_[0] = self.X[0]
        for n in range(self.T_free):
            self.run_one_step(n, np.zeros(self.Ni), 'freerun', online=False)
        self.x_init = self.X_[self.T_free - 1]

    def run_one_step(self, n, input, mode, online=False):
        if mode == 'freerun':
            x = self.X_[n, :]
            y = self.Y_[n, :]
        else:
            x = self.X[n, :]
            y = self.Y[n, :]

        sum = np.zeros(self.Nx)
        sum += self.Wr @ fx(x)
        # sum += self.Wi @ input.reshape(-1, 1)

        if reservoir.Ni == 1:
            sum += (self.Wi * input)[:, 0]
        else:
            sum += self.Wi @ input

        dx = (self.dt / self.tau) * (- x + sum)
        x += dx

        y = self.Wo @ f_kernel(x[:self.N_P])

        if online == True:
            q = self.P @ f_kernel(x)[:self.N_P]
            c = 1 / (1 + f_kernel(x)[:self.N_P].T @ q)
        error = y - input

        if mode == 'train' and online == True:
            self.mu = (n * self.mu + f_kernel(x)[:self.N_P]) / (n + 1)  # 平均
            self.P -= np.outer(q, q) * c

            self.Wo[:, :self.N_P] -= np.outer(error, q) * c

        if mode == 'test' and online == True:
            reservoir.TRAKR_errors[n] = np.linalg.norm(np.outer(error, q) * c) ** 2


        if mode == 'freerun':
            if n < self.T_free - 1:
                self.X_[n + 1, :] = x
                self.Y_[n + 1, :] = y
        else:
            if n < self.T1 - 1:
                self.X[n + 1, :] = x
                self.Y[n + 1, :] = y

class Windows:
    def __init__(self, param):
        self.T1 = param['T1']
        self.Nx = param['Nx']
        # self.Ny = param['Ny']
        self.Ni = param['Ni']
        self.N_P = param['N_P'] # window size
        self.train_length = param['train_length']

        self.sigma_noise = param['sigma_noise']
        self.alpha = param['alpha']

        ## 精度行列Pを用意
        self.P = np.identity(self.N_P) / self.alpha
        self.P_mod = np.zeros_like(self.P)
        self.mu = np.zeros(self.N_P)

    def sliding_windows(self, train_data):
        window_size = self.N_P
        self.X = np.zeros((len(train_data), window_size))
        self.X[window_size:, :] = [train_data[i - window_size:i] for i in range(window_size, len(train_data))]


def AnomalyDetect_SPE(data, reservoir, continual_learning=False, online=False):
    reservoir.P = np.identity(reservoir.N_P) / reservoir.alpha
    reservoir.P_mod = np.zeros_like(reservoir.P)
    reservoir.mu = np.zeros(reservoir.N_P)

    reservoir.reset(reservoir.x_init)

    for n in range(reservoir.T1):
        if continual_learning == False:
            if n < reservoir.train_length:
                reservoir.run_one_step(n, data[n], 'train', online=online)

                if n == reservoir.train_length - 1:
                    if online == True:
                        ## 精度行列を平均で補正
                        P_mu = reservoir.P @ reservoir.mu
                        tmp_mat = (reservoir.train_length / (
                                    1 - reservoir.train_length * reservoir.mu.T @ reservoir.P @ reservoir.mu)) * np.outer(P_mu,
                                                                                                                          P_mu)
                        reservoir.P_mod = reservoir.train_length * (reservoir.P + tmp_mat)  # 精度行列

                    elif online == False:
                        ## リードアウトを一括で学習
                        X_list = f_kernel(reservoir.X[:reservoir.train_length, :reservoir.N_P])
                        Y_list = data[:reservoir.train_length]
                        reservoir.Wo = Y_list.T @ np.linalg.pinv(X_list.T)  # Y_list.T = Wo @ X_list.Tを解く

            else:
                reservoir.run_one_step(n, data[n], 'test', online=False)


        elif continual_learning == True:
            reservoir.run_one_step(n, data[n], 'train', online=True)

    if reservoir.Ni == 1:
        SPE = (data - reservoir.Y[:, 0]) ** 2
    else:
        SPE = (data - reservoir.Y) ** 2
        SPE = np.sum(SPE, axis=1)
    SPE[:reservoir.train_length] = 0

    return SPE

def AnomalyDetect_TRAKR(data, reservoir, continual_learning, online):
    ## 精度行列を初期化
    reservoir.P = np.identity(reservoir.N_P) / reservoir.alpha

    TRAKR = np.zeros(reservoir.T1)
    reservoir.reset(reservoir.x_init)
    for n in range(reservoir.T1):
        if continual_learning == False:
            if n < reservoir.train_length:
                reservoir.run_one_step(n, data[n], 'train', online=online)

                if n == reservoir.train_length - 1:
                    if online == False:
                        ## 精度行列を一括で学習
                        x_mat = f_kernel(reservoir.X[:reservoir.train_length, :reservoir.N_P])
                        # cov_mat = (x_mat.T @ x_mat) / reservoir.train_length + reservoir.alpha * np.identity(reservoir.N_P)# 正則化あり
                        cov_mat = x_mat.T @ x_mat + reservoir.alpha * np.identity(reservoir.N_P)  # 正則化あり
                        reservoir.P = np.linalg.inv(cov_mat)
            else:
                reservoir.run_one_step(n, data[n], 'test', online=online)

        elif continual_learning == True:
            reservoir.run_one_step(n, data[n], 'train', online=True)

    TRAKR = reservoir.TRAKR_errors

    return TRAKR


def AnomalyDetect_MDRS(data, reservoir, precision_mode, continual_learning=False, online=False):
    ## 精度行列を初期化
    reservoir.P = np.identity(reservoir.N_P) / reservoir.alpha
    reservoir.P_mod = np.zeros_like(reservoir.P)
    reservoir.mu = np.zeros(reservoir.N_P)

    MDRS = np.zeros(reservoir.T1)
    reservoir.reset(reservoir.x_init)
    for n in range(reservoir.T1):
        if continual_learning == False:
            if n < reservoir.train_length:
                reservoir.run_one_step(n, data[n], 'train', online=online)

                if n == reservoir.train_length - 1:
                    if online == True:
                        ## 精度行列を平均で補正
                        P_mu = reservoir.P @ reservoir.mu
                        tmp_mat = (reservoir.train_length / (
                                    1 - reservoir.train_length * reservoir.mu.T @ reservoir.P @ reservoir.mu)) * np.outer(P_mu,
                                                                                                                          P_mu)
                        reservoir.P_mod = reservoir.train_length * (reservoir.P + tmp_mat)  # 精度行列

                    elif online == False:
                        ## 精度行列を一括で学習
                        x_mat = f_kernel(reservoir.X[:reservoir.train_length, :reservoir.N_P])
                        if precision_mode == 'zero':
                            # cov_mat = (x_mat.T @ x_mat) / reservoir.train_length + reservoir.alpha * np.identity(reservoir.N_P)# 正則化あり
                            cov_mat = x_mat.T @ x_mat + reservoir.alpha * np.identity(reservoir.N_P)  # 正則化あり
                            reservoir.P = np.linalg.inv(cov_mat)
                        elif precision_mode == 'mean':
                            reservoir.mu = np.mean(x_mat) # 平均
                            # cov_mat = np.cov(x_mat, rowvar=False) # 平均で補正あり
                            cov_mat = (x_mat - reservoir.mu).T @ (x_mat - reservoir.mu) + reservoir.alpha * np.identity(
                                reservoir.N_P)  # 平均で補正あり, 正則化あり
                            reservoir.P_mod = np.linalg.inv(cov_mat)


            else:
                reservoir.run_one_step(n, data[n], 'test', online=False)
                x = reservoir.X[n, :reservoir.N_P]
                if precision_mode == 'zero':
                    # print(x.shape, reservoir.P.shape)
                    MDRS[n] = f_kernel(x).T @ reservoir.P @ f_kernel(x)
                elif precision_mode == 'mean':
                    MDRS[n] = (f_kernel(x) - reservoir.mu).T @ reservoir.P_mod @ (
                            f_kernel(x) - reservoir.mu)  # 平均で補正

        elif continual_learning == True:
            reservoir.run_one_step(n, data[n], 'train', online=True)
            if n >= train_length:
                x = reservoir.X[n, :reservoir.N_P]
                MDRS[n] = f_kernel(x).T @ reservoir.P @ f_kernel(x)


    return MDRS

def AnomalyDetect_MDSW(data, reservoir, precision_mode, continual_learning=False, online=False):
    ## 精度行列を初期化
    reservoir.P = np.identity(reservoir.N_P) / reservoir.alpha
    reservoir.P_mod = np.zeros_like(reservoir.P)
    reservoir.mu = np.zeros(reservoir.N_P)

    MDSW = np.zeros(reservoir.T1)
    for n in range(reservoir.T1):
        if continual_learning == False:
            if n < reservoir.train_length:
                if n == reservoir.train_length - 1:
                    if online == True:
                        ## 精度行列を平均で補正
                        P_mu = reservoir.P @ reservoir.mu
                        tmp_mat = (reservoir.train_length / (
                                    1 - reservoir.train_length * reservoir.mu.T @ reservoir.P @ reservoir.mu)) * np.outer(P_mu,
                                                                                                                          P_mu)
                        reservoir.P_mod = reservoir.train_length * (reservoir.P + tmp_mat)  # 精度行列

                    elif online == False:
                        reservoir.sliding_windows(data[:reservoir.train_length])
                        ## 精度行列を一括で学習
                        if precision_mode == 'zero':
                            x_mat = f_kernel(reservoir.X[reservoir.N_P:reservoir.train_length, :reservoir.N_P])
                            # cov_mat = (x_mat.T @ x_mat) / reservoir.train_length + reservoir.alpha * np.identity(reservoir.N_P)# 正則化あり
                            cov_mat = x_mat.T @ x_mat + reservoir.alpha * np.identity(reservoir.N_P)  # 正則化あり
                            reservoir.P = np.linalg.inv(cov_mat)
                        elif precision_mode == 'mean':
                            reservoir.mu = np.mean(f_kernel(reservoir.X[reservoir.N_P:reservoir.train_length, :reservoir.N_P]))
                            cov_mat = np.cov(f_kernel(reservoir.X[:reservoir.train_length, :reservoir.N_P]), rowvar=False) # 平均で補正あり
                            reservoir.P_mod = np.linalg.inv(cov_mat) # 正則化なし


            else:
                x = data[n - reservoir.N_P:n] # スライド窓
                if precision_mode == 'zero':
                    MDSW[n] = f_kernel(x).T @ reservoir.P @ f_kernel(x)
                elif precision_mode == 'mean':
                    MDSW[n] = (f_kernel(x) - reservoir.mu).T @ reservoir.P_mod @ (
                            f_kernel(x) - reservoir.mu)  # 平均で補正


        elif continual_learning == True:
            reservoir.run_one_step(n, data[n], 'train', online=True)
            if n >= train_length:
                x = reservoir.X[n, :reservoir.N_P]
                MDSW[n] = f_kernel(x).T @ reservoir.P @ f_kernel(x)
    return MDSW



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run anomaly detection with specified benchmark and method.')
    parser.add_argument('--benchmark', type=str, required=True,
                        help='Benchmark dataset to use (e.g., UCR, SMD, SWaT)')
    parser.add_argument('--method', type=str, required=True,
                        help='Method to use for anomaly detection (e.g., SPE, TRAKR, MDRS, MDSW_fixed, MDSW_adjusted)')

    # Parse the arguments
    args = parser.parse_args()

    # Use command-line arguments
    benchmark = args.benchmark
    method = args.method

    param = {
        'T1': 10000,  # 時間サイズ
        'Nx': 500,  # ネットワークのノード数
        'N_P': 200,  # 精度行列のサイズ
        'dt': 0.01,  # タイムステップ
        'sigma_noise': 0,  # ノイズの標準偏差
        'alpha': 1e-7,  # 正則化係数
        'alpha_r': 1.3,  # リカレント強度
        'beta_r': 1.0,  # リカレント非ゼロ率
        'alpha_i': 1.0,  # 入力ベクトルの強度
        'beta_i': 0.25,  # 入力ベクトルの非ゼロ率
        'alpha_b': 1.0,  # バイアスの強度
        'beta_b': 0.25,  # バイアスの非ゼロ率
    }

    # タイムコンスタントの設定
    mix_ratio = 0.9
    param['tau'] = np.zeros(param['Nx'])
    param['tau'][:int(param['Nx'] * (1 - mix_ratio))] = 0.01
    param['tau'][int(param['Nx'] * (1 - mix_ratio)):] = 0.025

    # 乱数シードの設定
    np.random.seed(0)


    benchmark_path = f'benchmarks/{benchmark}'

    folder_path = f'{benchmark_path}/data'
    result_path = f'{benchmark_path}/results'
    plot_path = f'{benchmark_path}/plots'
    analysis_path = f'{benchmark_path}/analysis'

    # Create directories if they do not exist
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(analysis_path, exist_ok=True)

    # continual_learning = True
    continual_learning = False

    if method == 'TRAKR':
        online = True
    else:
        online = False

    runtime_list = []

    if benchmark in ['UCR']:
        filename_list = os.listdir(folder_path)
    elif benchmark in ['SMD', 'SMAP', 'MSL']:
        filename_list = os.listdir(f'{folder_path}/test_label')
    elif benchmark in ['PSM', 'SWaT', 'MSL_combined', 'SMAP_combined', 'SMD_combined']:
        filename_list = [benchmark] #  1つしかデータセットがないため
    elif benchmark in ['WADI']:
        filename_list = os.listdir(f'{folder_path}')

    for filename in filename_list:
        print(filename)

        ## データ成形
        if benchmark in ['UCR']:
            match = re.search(r"(\d+)_(\d+)_(\d+)\.txt$", filename)
            # Load time series data
            data = np.loadtxt(os.path.join(folder_path, filename))
            # Extract parameters from filename
            train_length, anomaly_start, anomaly_end = map(int, match.groups())
            anomaly_length = anomaly_end - anomaly_start + 1
            param['T1'] = len(data)
            param['train_length'] = train_length
            param['Ni'] = 1
            param['Ny'] = 1

            # Create anomaly labels for test data
            test_data_length = len(data) - train_length
            anomaly_label = np.zeros(test_data_length)
            anomaly_label[anomaly_start - train_length: anomaly_end - train_length + 1] = 1

        else:
            if benchmark in ['SMD']:
                data_train = np.loadtxt(f'{folder_path}/train/{filename}', delimiter=',')
                data_test = np.loadtxt(f'{folder_path}/test/{filename}', delimiter=',')
                anomaly_label = np.loadtxt(f'{folder_path}/test_label/{filename}')
            elif benchmark in ['SMAP', 'MSL']:
                filename_npy = filename.replace('.txt', '.npy')
                data_train = np.load(f'{folder_path}/train/{filename_npy}')
                data_test = np.load(f'{folder_path}/test/{filename_npy}')
                anomaly_label = np.loadtxt(f'{folder_path}/test_label/{filename}')
            elif benchmark in ['PSM']:
                df_train = pd.read_csv(f'{folder_path}/train.csv')
                df_test = pd.read_csv(f'{folder_path}/test.csv')
                df_anomaly_label = pd.read_csv(f'{folder_path}/test_label.csv')
                data_train = df_train.iloc[:, 1:].to_numpy()
                data_test = df_test.iloc[:, 1:].to_numpy()
                anomaly_label = df_anomaly_label.iloc[:, 1:].to_numpy().flatten()
            elif benchmark in ['SWaT']:
                # df_train = pd.read_csv(f'{folder_path}/SWaT_Dataset_Normal_v0.csv')
                df_train = pd.read_csv(f'{folder_path}/SWaT_Dataset_Normal_v1.csv')
                data_train = df_train.iloc[:, 1:-1].to_numpy() # 1列目と最後の列を除外
                df_test = pd.read_csv(f'{folder_path}/SWaT_Dataset_Attack_v0.csv')
                anomaly_label = df_test.iloc[:, -1].to_numpy().flatten()
                anomaly_label = np.where(anomaly_label == 'Normal', 0, 1)
                data_test = df_test.iloc[:, 1:-1].to_numpy()  # 1列目と最後の列を除外

            elif benchmark in ['MSL_combined', 'SMAP_combined', 'SMD_combined']:
                benchmark_name = benchmark.replace('_combined', '')
                data_train = np.load(f'{folder_path}/{benchmark_name}_train.npy')
                data_test = np.load(f'{folder_path}/{benchmark_name}_test.npy')
                anomaly_label = np.load(f'{folder_path}/{benchmark_name}_test_label.npy')

            data = np.concatenate((data_train, data_test), axis=0)
            print('data shape:', data.shape)
            ## 欠損値をchannelの平均値で補完
            if benchmark in ['PSM']:
                # 列の平均値を計算（NaNを無視）
                means = np.nanmean(data, axis=0)
                # NaNを列の平均値で置換
                data = np.where(np.isnan(data), means, data)

            param['train_length'] = data_train.shape[0]
            param['T1'] = data.shape[0]
            param['Ni'] = data.shape[1]
            param['Ny'] = data.shape[1]

            # 異常区間の開始点と終了点を検出
            start_end = np.where(np.diff(anomaly_label) != 0)[0] + 1
            if anomaly_label[0] == 1:  # 最初の要素が1ならば開始点をリストに追加
                start_end = np.insert(start_end, 0, 0)
            if anomaly_label[-1] == 1:  # 最後の要素が1ならば終了点をリストに追加
                start_end = np.append(start_end, len(anomaly_label))


        ## 列ごと（チャンネルごと）に正規化
        channel_min = np.min(data, axis=0)  # 各チャンネルの最小値
        channel_max = np.max(data, axis=0)  # 各チャンネルの最大値
        normalized_data = (data - channel_min) / (channel_max - channel_min + 1e-10) # 分母ゼロを避ける

        start_time = time.time()
        if method in ['SPE', 'TRAKR', 'MDRS']:
            ## リザバー生成
            reservoir = Reservoir(param)
            reservoir.generate_weight_matrices()
            ## リザバーの初期値設定
            reservoir.freerun()

            normalized_data = 2.0 * normalized_data + 2.0
            ## チャンネルの数で正規化(必要)
            normalized_data /= param['Ni']

        elif method in ['MDSW_fixed', 'MDSW_adjusted']:
            reservoir = Windows(param)

            # Load the provided CSV file
            file_path = 'benchmarks/UCR/DAMP_results.csv'
            data = pd.read_csv(file_path)

            # Create a dictionary mapping file names to their SubsequenceLength
            subsequence_length_dict = pd.Series(data.SubsequenceLength.values, index=data.FileName).to_dict()


        if method == 'MDSW_adjusted':
            reservoir.N_P = subsequence_length_dict.get(filename, None)  # 疑似周期をDAMPの結果から取得

        if method == 'SPE':
            anomaly_score = AnomalyDetect_SPE(normalized_data, reservoir, continual_learning=continual_learning,
                                    online=online)
        elif method == 'TRAKR':
            anomaly_score = AnomalyDetect_TRAKR(normalized_data, reservoir, continual_learning=continual_learning, online=online)
        elif method == 'MDRS':
            anomaly_score = AnomalyDetect_MDRS(normalized_data, reservoir, precision_mode='mean', continual_learning=continual_learning, online=online)
        elif method in ['MDSW_fixed',  'MDSW_adjusted']:
            anomaly_score = AnomalyDetect_MDSW(normalized_data, reservoir, precision_mode='mean', continual_learning=continual_learning, online=online)
        end_time = time.time()


        filename = os.path.splitext(filename)[0]
        np.savetxt(f'{result_path}/{filename}_{method}.txt', anomaly_score)


        elapsed_time = end_time - start_time
        print('runtime:', elapsed_time)
        runtime_list.append(elapsed_time)

        fig = plt.figure()
        ax = fig.add_subplot(211)
        if benchmark == 'UCR':
            plt.axvspan(anomaly_start, anomaly_end, color='tab:orange', alpha=0.5,
                        linewidth=0)
        else:
            for i in range(0, len(start_end), 2):
                plt.axvspan(reservoir.train_length + start_end[i], reservoir.train_length + start_end[i + 1], color='tab:orange', alpha=0.5, linewidth=0)
        ax.plot(normalized_data, linewidth=0.8)
        plt.title(filename)

        ax = fig.add_subplot(212, sharex=ax)
        if benchmark == 'UCR':
            plt.axvspan(anomaly_start, anomaly_end, color='tab:orange', alpha=0.5,
                        linewidth=0)
        else:
            for i in range(0, len(start_end), 2):
                plt.axvspan(reservoir.train_length + start_end[i], reservoir.train_length + start_end[i + 1], color='tab:orange', alpha=0.5, linewidth=0)

        ax.plot(anomaly_score, linewidth=0.8)

        plt.savefig(f'{plot_path}/{filename}_{method}.png')
        plt.close()

    ## save runtime
    time_path = f'{benchmark_path}/analysis/runtime_{method}.csv'
    runtime_df = pd.DataFrame({'Filename': filename_list, 'Runtime': runtime_list})
    runtime_df = runtime_df.sort_values(by='Filename')
    runtime_df.to_csv(time_path, index=False)



