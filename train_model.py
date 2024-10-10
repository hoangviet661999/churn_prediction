import pandas as pd
import argparse
from churn_prediction.dataset import process_data
from churn_prediction.model import train_model, eval_model
from sklearn.model_selection import train_test_split
import numpy as np
from omegaconf import OmegaConf
import hydra

def main(args, config):
    dataset = pd.read_csv(args.csv_path)

    features = [
        'CreditScore',
        'Geography',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary'
    ]
    X = dataset[features]
    y = np.array(dataset['Exited'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    X_train, data_pipeline = process_data(X_train, training=True)
    X_val, _ = process_data(X_val, training=False, data_pipeline=data_pipeline)

    param = {
        'C' : config['hyperparameters']['C'],
        'penalty' : config['hyperparameters']['penalty'],
        'solver' : config['hyperparameters']['solver']
    }

    model = train_model(X_train, y_train, **param)
    precision, recall, f1 = eval_model(model, X_val, y_val)
    print(f"{precision =}\n{recall =}\n{f1 =}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=None, help = 'path to csv path')
    parser.add_argument('--cfg_path', type=str, default='configs/config.yaml', help='path to config path')
    args = parser.parse_args()
    config = OmegaConf.load(args.cfg_path)
    main(args, config)