import numpy as np
import yaml
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
from data_loader import MAPDataset
from preprocess import read_load_trace_data, preprocessing

batch_size = 256

def find_optimal_threshold(train_df, predicted_column='past', target_column='future', num_samples=None):
    y_real = np.array(train_df[target_column].values)
    y_score = train_df[predicted_column].values

    num_future_samples = len(train_df)

    if num_samples is not None:
        num_samples = min(num_samples, num_future_samples)
        sampled_idx_future = np.random.choice(num_future_samples, num_samples, replace=False)
        sampled_idx_past = np.repeat(sampled_idx_future, y_score.shape[1])
        y_real = np.repeat(y_real[sampled_idx_future], y_score.shape[1])
        y_score = np.array([y_score[i] for i in sampled_idx_past])
    y_score = y_score.reshape(-1, 1)
    y_pred_bin = np.array([1 if row_score >= 0.5 else 0 for row_score in y_score])

    print("y_real shape:", y_real.shape)
    print("y_pred_bin shape:", y_pred_bin.shape)

    micro_f1 = f1_score(y_real, y_pred_bin, average='micro')

    return micro_f1

def data_generator(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM,only_val=False):
    with open('params.yaml', 'r') as p:
        params = yaml.safe_load(p)
    hardware = params['hardware']
    if only_val==True:
        print("only validation")
        _, eval_data = read_load_trace_data(file_path, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
        df_test = preprocessing(eval_data, hardware)
        test_dataset = MAPDataset(df_test)

        #logging.info("-------- Dataset Build! --------")
        dev_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)

        return dev_dataloader, df_test
    else:
        print("train and validation")
        train_data, eval_data = read_load_trace_data(file_path, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
        df_train = preprocessing(train_data, hardware)
        df_test = preprocessing(eval_data, hardware)

        train_dataset = MAPDataset(df_train)
        test_dataset = MAPDataset(df_test)

        #logging.info("-------- Dataset Build! --------")
        train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
        dev_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)

        return train_dataloader, dev_dataloader, df_train,df_test

train_loader, test_loader, df_train, df_test = data_generator('/data/pengmiao/ML-DPC-S0/LoadTraces/473.astar-s0.txt.xz', 1, 2, 0)

print('df train shape',df_train.shape)
print('df train past size',df_train['past'].size)
print('df train past shape', df_train['past'].shape)
print('df train future size', df_train['future'].size)
print('df train future shape',df_train['future'].shape)
print(find_optimal_threshold(df_train))
