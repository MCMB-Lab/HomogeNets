import torch
import pandas as pd


def module_to_torchscript(module, fname):
    """!
    https://pytorch.org/tutorials/advanced/cpp_export.html
    """
    module.train(False)
    traced_script_module = torch.jit.script(module)
    traced_script_module.save(fname)


class CSVDataset(torch.utils.data.Dataset):
    """!
    Reads a csv of type:
    [Header row]
    x00 x01 x02 ... y00 y01 y02 ...
    x10 x11 x12 ... y10 y11 y12 ...
    x20 x21 x22 ... y20 y21 y22 ...
    x30 x31 x32 ... y30 y31 y32 ...
    {_inputs_______}{__outputs_____}
                     ^
                     |
                     target_col

    @param csv_file the file location for the csv file
    @param target_col the column index for the start of the outputs
    @param data_range a tuple for the range of rows in this dataset e.g. (0, 10)
    will select the first 10 rows of the data
    """

    def __init__(
        self, csv_file, data_cols, input_cols, target_cols, data_range, seed=None
    ):
        # pandas dataframe automatically deals with header file and parses data
        csv_data_frame = pd.read_csv(csv_file)
        self.data = torch.tensor(csv_data_frame.values)
        if seed is not None:
            torch.manual_seed(seed)
            indices = torch.randperm(self.data.shape[0])
            self.data = self.data[indices]
        self.data = self.data[
            data_range[0] : data_range[1], data_cols[0] : data_cols[1]
        ]
        self.input_cols = input_cols
        self.target_cols = target_cols

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        inputs = self.data[idx, self.input_cols[0] : self.input_cols[1]]
        outputs = self.data[idx, self.target_cols[0] : self.target_cols[1]]
        return inputs, outputs


def z_score(data, cols):
    mu = torch.mean(data[:, cols[0] : cols[1]], 0)
    sigma = torch.std(data[:, cols[0] : cols[1]], 0)
    return mu, sigma


def maxmin(data, cols):
    maxs = torch.max(data[:, cols[0] : cols[1]], 0).values
    mins = torch.min(data[:, cols[0] : cols[1]], 0).values
    return maxs, mins


def load_data(fname_data, num_data, data_cols, input_cols, output_cols, seed=None):
    num_train_data = int(num_data * 2 / 3)
    train_data = CSVDataset(
        fname_data, data_cols, input_cols, output_cols, (0, num_train_data), seed
    )
    val_data = CSVDataset(
        fname_data, data_cols, input_cols, output_cols, (num_train_data, num_data), seed
    )

    return train_data, val_data
