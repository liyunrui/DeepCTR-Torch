import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("../")
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import argparse

def train_test_split_by_time(data, time_field="str_fixture_date", test_size=0.2):
    data[time_field] = pd.to_datetime(data[time_field])
    data.sort_values(by = time_field, inplace=True)
    n = len(data)
    first_top = int(n * (1-test_size))
    train, test = data[:first_top], data[first_top:]
    return train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='movielens_sample')
    parser.add_argument('--linear_only', type=bool, default=False)
    parser.add_argument('--time_split', type=bool, default=False)
    args, _ = parser.parse_known_args()
    if args.dataset == "movielens_sample":
        data = pd.read_csv("./movielens_sample.txt")
        sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip"]
        target = ['rating']
    elif args.dataset == "dazn":
        data = pd.read_csv("./dazn_sample.txt")
        sparse_features = ["str_fixture_id","str_outlet","str_competition_name","str_sport_name"]
        target = ["engagement_rate"]
        print(f"loading  {args.dataset}")
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    if args.time_split:
        print("time split validation ..")
        train, test = train_test_split_by_time(data, test_size=0.2)
    else:
        train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device, linear_only=args.linear_only, l2_reg_linear=1e-2)
    model.compile("adam", "mse", metrics=['mse',"mae"])

    history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=100, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test MSE", round(mean_squared_error(
        test[target].values, pred_ans), 4))
