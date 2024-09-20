import argparse
import numpy as np
import yaml
import time
import torch
import pickle as pkl
import datetime
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as Data
from sklearn.manifold import TSNE
from torch_geometric.graphgym import optim

from lib.dataloader import normal_and_generate_dataset_time, get_mask, get_adjacent, get_grid_node_map_maxtrix
from model.STACC import STACC
from lib.utils import mask_loss, compute_loss, predict_and_evaluate, load_graph
from model.trainer import Trainer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import os

torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print('CUDA_LAUNCH_BLOCKING')
print(os.getenv('CUDA_LAUNCH_BLOCKING'))


def graph_y_after_dot(data, label="Label stats", dim=1):

    if dim >= data.ndim:
        print(f"Error: dim parameter {dim} is out of bounds for data with {data.ndim} dimensions")
        return

    if data.ndim == 3:
        data_dim = data[:, :, :] if dim == 2 else data[:, :, dim] if data.shape[1] == 1 else data[:, dim, :]
    else:
        print("Error: Unsupported data dimensions")
        return

    unique_values = np.unique(data_dim)
    non_zero_count = np.count_nonzero(data_dim)

    print("Unique values in dimension:", unique_values)
    print("Number of non-zero entries in dimension:", non_zero_count)



def main(args):
    loaders = []
    scaler = ""
    train_data_shape = ""
    graph_feature_shape = ""

    start_date = datetime.datetime(2013, 1, 1)
    target_start_date = datetime.datetime(2013, 11, 17)
    target_end_date = datetime.datetime(2013, 11, 24)


    start_index = (target_start_date - start_date).days * 24
    end_index = (target_end_date - start_date).days * 24 + 24

    for idx, (x, y, target_times, scaler) in enumerate(
            normal_and_generate_dataset_time(
                args.all_data_filename,
                args.train_rate,
                args.valid_rate,
                args.recent_prior,
                args.week_prior,
                args.one_day_period,
                args.days_of_week,
                args.pre_len,
                test_start_index=None,
                test_end_index=None
            )):

        if args.test:
            x = x[:100]
            y = y[:100]
            target_times = target_times[:100]

        if 'nyc' in args.all_data_filename:
            # print("graph_x shape:", x.shape)
            # print("graph_y shape:", y.shape)

            mamba_x = x[:, :, 0, :, :].reshape(
                (x.shape[0], x.shape[1], args.north_south_map * args.west_east_map))
            graph_x = x[:, :, [0, 46, 47], :, :].reshape(
                (x.shape[0], x.shape[1], -1, args.north_south_map * args.west_east_map))
            graph_y = y[:, :, :, :].reshape((y.shape[0], y.shape[1], args.north_south_map * args.west_east_map))


            mamba_x = np.dot(mamba_x, grid_node_map)
            graph_x = np.dot(graph_x, grid_node_map)



            graph_y = np.dot(graph_y, grid_node_map)

            graph_y_after_dot(graph_y, "Stats of graph_y after dot product", dim=2)

        if 'chicago' in args.all_data_filename:
            mamba_x = x[:, :, 0, :, :].reshape(
                (x.shape[0], x.shape[1], args.north_south_map * args.west_east_map))
            graph_x = x[:, :, [0, 39, 40], :, :].reshape(
                (x.shape[0], x.shape[1], -1, args.north_south_map * args.west_east_map))
            graph_x = np.dot(graph_x, grid_node_map)
            graph_y = y[:, :, :, :].reshape((y.shape[0], y.shape[1], args.north_south_map * args.west_east_map))
            graph_y = np.dot(graph_y, grid_node_map)
            mamba_x = np.dot(mamba_x, grid_node_map)
        print("feature:", str(x.shape), "label:", str(y.shape), "time:", str(target_times.shape))

        if idx == 0:
            scaler = scaler  # -- record max and min
            train_data_shape = x.shape
            time_shape = target_times.shape
            graph_feature_shape = graph_x.shape

        graph_x_tensor = torch.from_numpy(graph_x)
        mamba_x_tensor = torch.from_numpy(mamba_x)
        graph_y_tensor = torch.from_numpy(graph_y)
        target_times_tensor = torch.from_numpy(target_times)


        loaders.append(Data.DataLoader(
            Data.TensorDataset(

                torch.from_numpy(graph_x),
                torch.from_numpy(graph_y),
                torch.from_numpy(target_times),
                torch.from_numpy(mamba_x),

            ),
            args.batch_size,
            shuffle=(idx == 0)
        ))

    # train_loader, val_loader, test_loader = loaders

    torch.backends.cudnn.enabled = False

    graph = load_graph(args.graph_file, device=args.device)


    risk_mask = get_mask("data/nyc/risk_mask.pkl")  # 20*20

    risk_mask_flat = risk_mask.reshape(-1)
    risk_mask = np.dot(risk_mask_flat, grid_node_map)


    args.num_nodes = len(graph)  # （243,243）
    model = STACC(args).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    actual_trainable_params = count_parameters(model)
    print(f"Actual trainable parameters during training: {actual_trainable_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr_init)



    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        graph=graph,
        scaler=scaler,
        args=args,
        risk_mask=risk_mask
    )
    results = None

    if args.mode == 'train':
        results = trainer.train()
    elif args.mode == 'test':
        # test
        state_dict = torch.load(
            args.best_path,
            map_location=torch.device(args.device)
        )
        model.load_state_dict(state_dict['model'])
        print("Load saved model")
        results = trainer.test(model, loaders[1], [scaler],
                               graph, trainer.logger, trainer.args)


    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='config/nyc/NYCTaxi.yaml',
                        type=str, help='the configuration to use')

    args = parser.parse_args()

    print(f'Starting experiment with configurations in {args.config_filename}...')
    time.sleep(3)
    configs = yaml.load(  # v
        open(args.config_filename),
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**configs)
    grid_node_map = get_grid_node_map_maxtrix(args.grid_node_filename)
    main(args)
