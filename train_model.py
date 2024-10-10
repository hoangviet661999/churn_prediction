import pandas as pd
import argparse
from churn_prediction.dataset import process_data
from churn_prediction.model import train_model, eval_model
from sklearn.model_selection import train_test_split
import numpy as np

def main(args):
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
        'C' : .422,
        'penalty' : "l1",
        'solver' : "saga"
    }

    model = train_model(X_train, y_train, **param)
    precision, recall, f1 = eval_model(model, X_val, y_val)
    print(f"{precision =}\n{recall =}\n{f1 =}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=None, help = 'path to csv path')
    args = parser.parse_args()
    main(args)