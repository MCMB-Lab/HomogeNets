#test
import torch

from homogenets.surrogate import (
    NNwithJacobian,
    NN,
    ModelAndJacobianTrainer,
    ModelTrainer,
    RL2Loss,
    RMSELoss,
    RelLoss,
    AngleLoss,
    load_data,
    z_score,
    maxmin,
    module_to_torchscript,
)
from homogenets.surrogate import (
    SigmoidActivation,
    TanhActivation,
    ZscoreActivation,
    DeZscoreActivation,
    MaxActivation,
    DeMaxActivation,
)


def diag_count(n):
    x = 0
    for i in range(n + 1):
        x += i
    return x


DVAR_ELLIPSE = 3
DVAR_FRAME = 1
DVAR_HSTRUSS = 1
DVAR_OCTET = 1

GRAD_DIM2D = 4
DGRAD_DIM2D = 8
CDIAG_2D = diag_count(4)  # 4 + 3 + 2 + 1
QDIAG_2D = diag_count(8)

GRAD_DIM3D = 9
DGRAD_DIM3D = 27
CDIAG_3D = diag_count(9)
QDIAG_3D = diag_count(27)


def train_linearNN(
    save_dir: str,
    fname_data: str,
    num_data: int = 1000,  # number of data to train with
    inp: int = 1,
    input_cols: tuple[int] = [0, 4],
    output_cols: tuple[int] = [0, 4],
    seed: int = 1996,
    batch_size: int = 8,
    hidden_size: int = 32,
    num_hidden_layers: int = 3,
    num_epochs: int = 1000,
    lr: float = 1.0e-2,
):
    # DATA
    train_data, val_data = load_data(
        fname_data,
        num_data,
        (input_cols[0], output_cols[1]),
        input_cols,
        output_cols,
        seed,
    )

    # prune data
    MAXVAL = 10.0
    maxs = torch.max(torch.abs(val_data.data), 1).values
    print(val_data.data.shape)
    val_data.data = val_data.data[maxs < MAXVAL]
    print(val_data.data.shape)

    maxs = torch.max(torch.abs(train_data.data), 1).values
    print(train_data.data.shape)
    train_data.data = train_data.data[maxs < MAXVAL]
    print(train_data.data.shape)

    # standardization
    x_means, x_stds = z_score(train_data.data, input_cols)
    y_means, y_stds = z_score(train_data.data, output_cols)

    x_maxs, x_mins = maxmin(train_data.data, input_cols)
    y_maxs, y_mins = maxmin(train_data.data, output_cols)

    # MODEL
    model = NN(
        seed,
        input_cols[1] - input_cols[0],
        output_cols[1] - output_cols[0],
        batch_size,
        hidden_size,
        num_hidden_layers,
        activation_func=SigmoidActivation,
        # activation_func=SigmoidActivation,
        # activation_func=ReLUActivation,
        # pre_processor=ZscoreActivation,
        # post_processor=DeZscoreActivation,
        # pre_processor_args=(x_means, x_stds),
        # post_processor_args=(y_means, y_stds),
        pre_processor=MaxActivation,
        post_processor=DeMaxActivation,
        pre_processor_args=(x_maxs, x_mins),
        post_processor_args=(y_maxs, y_mins),
    )

    # TRAIN
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # loss_fn = torch.nn.MSELoss() # mean( (y-yhat)^T (y-yhat) )
    loss_fn = torch.nn.MSELoss(reduction="sum")
    # loss_fn = torch.nn.L1Loss()
    # loss_fn = torch.nn.SmoothL1Loss()

    log_every_X = 10
    trainer = ModelTrainer(model, train_dataloader, val_dataloader, optimizer, loss_fn)
    trainer.train(save_dir, num_epochs, log_every_X)

    # SAVE
    module_to_torchscript(trainer.model, save_dir + ".pt")


def train_NN(
    save_dir: str,
    fname_data: str,
    num_data: int = 1000,  # number of data to train with
    num_design_vars: int = 0,  # number of parameters in microstructure
    grad_size: int = 4,  # size of gradu
    Dgrad_size: int = 0,  # 0 if firstorder dataset, else the size of the gradgradu
    C_size: int = 4 * 4,  # size of firstordertangent
    Q_size: int = 8 * 8,  # size of secondordertangent
    train_secondorder: bool = False,  # train for "Q" and "Qmat", else "P" and "Cmat"
    train_jacobian: bool = False,  # train the backprop of the net to meet the actual jacobian
    train_jac_fwd: bool = False,  # train the net to predict a jacobian
    jacobian_scale: int = 1,  # divide the jacobian loss by this value
    seed: int = 1996,
    batch_size: int = 8,
    hidden_size: int = 32,
    num_hidden_layers: int = 3,
    num_epochs: int = 1000,
    loss: str = "MSE",
    lr: float = 1.0e-3,
    log_every_X: int = 10,
    schedule: bool = False,
    noise: float = 0.0,
):
    # sizes of the dataset
    # {design vars, grad, Dgrad} -> {P, Cmat, Q, Qmat}
    data_input_size = num_design_vars + grad_size + Dgrad_size
    data_output_size = grad_size + C_size + Dgrad_size + Q_size

    # output_cols: the outputs of interest relative to the global dataaset indices
    # dim_fwd: the output cols of interest relative to the "y" vector of size data_output_size
    # data_jac_fwd: the output cols corresponding to the jacobian relative to the "y" vector of size data_output_size
    # dim_jac_in: input cols corresponding to jacobian relative to the "x" vector of size data_input_size
    # dim_jac_out: output cols corresponding to jacobian
    if train_secondorder:
        dim_fwd = (0, Dgrad_size)
        data_jac_fwd = (Dgrad_size, Dgrad_size + Q_size)
        dim_jac_in = (
            num_design_vars + grad_size,
            num_design_vars + grad_size + Dgrad_size,
        )
        dim_jac_out = (0, Dgrad_size)
        output_cols = (
            data_input_size + grad_size + C_size,
            data_input_size + data_output_size,
        )
    else:
        dim_fwd = (0, grad_size)
        data_jac_fwd = (grad_size, grad_size + C_size)
        dim_jac_in = (num_design_vars, num_design_vars + grad_size)
        dim_jac_out = (0, grad_size)
        output_cols = (data_input_size, data_input_size + grad_size + C_size)

    # if True, the NN tries to predict the jacobian directly
    if train_jac_fwd:
        dim_jac_fwd = data_jac_fwd
        if train_secondorder:
            model_output_size = Dgrad_size + Q_size
        else:
            model_output_size = grad_size + C_size
    else:
        dim_jac_fwd = None
        if train_secondorder:
            model_output_size = Dgrad_size
        else:
            model_output_size = grad_size

    # DATA
    train_data, val_data = load_data(
        fname_data,
        num_data,
        (0, data_input_size + data_output_size),  # load all columns of data
        (0, data_input_size),
        output_cols,
        seed,
    )

    # prune data
    MAXVAL = 10.0
    maxs = torch.max(torch.abs(val_data.data), 1).values
    print(val_data.data.shape)
    val_data.data = val_data.data[maxs < MAXVAL]
    print(val_data.data.shape)

    maxs = torch.max(torch.abs(train_data.data), 1).values
    print(train_data.data.shape)
    train_data.data = train_data.data[maxs < MAXVAL]
    print(train_data.data.shape)

    # standardization
    x_means, x_stds = z_score(train_data.data, (0, data_input_size))
    y_means, y_stds = z_score(
        train_data.data, (output_cols[0], output_cols[0] + model_output_size)
    )

    x_maxs, x_mins = maxmin(train_data.data, (0, data_input_size))
    y_maxs, y_mins = maxmin(
        train_data.data, (output_cols[0], output_cols[0] + model_output_size)
    )

    # MODEL
    model = NNwithJacobian(
        seed,
        data_input_size,
        model_output_size,
        batch_size,
        hidden_size,
        num_hidden_layers,
        dim_fwd=dim_fwd,
        dim_jac_in=dim_jac_in,
        dim_jac_out=dim_jac_out,
        dim_jac_fwd=dim_jac_fwd,
        batch_norm=False,
        layer_norm=False,
        dropout=False,
        # activation_func=SigmoidActivation,
        # activation_func=IdentityActivation,
        # activation_func=ReLUActivation,
        activation_func=TanhActivation,
        pre_processor=ZscoreActivation,
        post_processor=DeZscoreActivation,
        # pre_processor=SafeZscoreActivation,
        # post_processor=SafeDeZscoreActivation,
        pre_processor_args=(x_means, x_stds),
        post_processor_args=(y_means, y_stds),
        # pre_processor=MaxActivation,
        # post_processor=DeMaxActivation,
        # pre_processor=SafeMaxActivation,
        # post_processor=SafeDeMaxActivation,
        # pre_processor_args=(x_maxs, x_mins),
        # post_processor_args=(y_maxs, y_mins),
    )

    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()

    print(f"Number of parameters: {total_params}")

    # TRAIN
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn="strong_wolfe")
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if loss == "MSE":
        # loss_fn = torch.nn.MSELoss() # mean( (y-yhat)^T (y-yhat) )
        loss_fn = torch.nn.MSELoss(reduction="sum")
        # loss_fn = torch.nn.L1Loss(reduction='sum')
        # loss_fn = torch.nn.SmoothL1Loss()
    elif loss == "RL1":
        loss_fn = RL2Loss(order=1)
    elif loss == "RL2":
        loss_fn = RL2Loss(order=2)
    elif loss == "RMSE":
        loss_fn = RMSELoss()
    elif loss == "Rel":
        loss_fn = RelLoss()
    elif loss == "Angle":
        loss_fn = AngleLoss()
    else:
        raise Exception("Loss function not recognized: " + loss)

    trainer = ModelAndJacobianTrainer(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        data_jac_fwd,
        train_fwd=True,
        train_jacobian=train_jacobian,
        jacobian_scale=jacobian_scale,
        noise=noise,
        schedule=schedule,
    )
    trainer.train(save_dir, num_epochs, log_every_X)
    trainer.model.eval()

    # TEST
    verbose = False
    x, y = next(iter(trainer.train_dataloader))
    # x = x[0:1,...]
    # y = y[0:1,...]
    yhat = trainer.model(x)
    if verbose:
        print("\nTrain example:")
        print(x)
        print(y)
        print(yhat)

    yhat = trainer.model(x)
    yhat_i, dyhat_dx = trainer.model.jacobian_from_fwd(yhat)
    jachat = trainer.model.jacobian_from_bwd(yhat)
    y_i, jac = trainer.model.jacobian_from_fwd(y, trainer.dim_jac_fwd)

    if verbose:
        print("x")
        print(x)
        print("y")
        print(y)
        print("y_i")
        print(y_i)
        print(yhat_i)
        print("jacobian")
        print(jac)
        print(dyhat_dx)
        print(jachat)

    x, y = next(iter(trainer.val_dataloader))
    x = x[0:1, ...]
    y = y[0:1, ...]
    yhat = trainer.model(x)
    if verbose:
        print("\nTest example:")
        print(x)
        print(y)
        print(yhat)

    # test
    jac = trainer.model.jacobian(x)
    jac_FD = trainer.model.jacobian_by_FD(x, 1.0e-8)
    jac_bwd = trainer.model.backward(yhat)

    if verbose:
        print("\nJacobian:", jac)
        print("\nJacobian FD:", jac_FD)
    max_err = torch.max(torch.abs((jac_FD - jac) / jac))
    print("FD error = {:.4e}".format(max_err.item()))
    if verbose:
        print("\nJacobian BWD:", jac_bwd)
    max_err = torch.max(torch.abs((jac_bwd - jac) / jac))
    print("BWD error = {:.4e}".format(max_err.item()))

    if verbose:
        y_i, jac = trainer.model.jacobian_from_fwd(y, trainer.dim_jac_fwd)
        print(jac)
        dyhatdx = trainer.model.backward(yhat)
        print(dyhatdx)

        dyhatdx = trainer.model.jacobian_from_bwd(yhat)
        print(dyhatdx)

    # SAVE
    module_to_torchscript(trainer.model, save_dir + ".pt")


def train_linear_firstorder_ellipse():
    train_linearNN(
        "./results/train_linear_firstorder_ellipse",
        "./data/generate_linear_firstorder_ellipse.csv",
        num_data=1000,
        input_cols=(0, DVAR_ELLIPSE),
        output_cols=(DVAR_ELLIPSE, DVAR_ELLIPSE + CDIAG_2D),
        num_epochs=20,
    )


def train_nonlinear_firstorder_ellipse():
    train_NN(
        "./results/train_nonlinear_firstorder_ellipse",
        "./data/generate_nonlinear_firstorder_ellipse.csv",
        num_data=10_000,
        num_design_vars=DVAR_ELLIPSE,
        grad_size=GRAD_DIM2D,
        Dgrad_size=0,
        C_size=GRAD_DIM2D * GRAD_DIM2D,
        Q_size=0,
        num_epochs=20,
        train_secondorder=False,
        train_jacobian=True,
        jacobian_scale=0.01,
        train_jac_fwd=False,
        loss="RL2",
        lr=1.0e-3,
        schedule=True,
    )


def main():
    train_linear_firstorder_ellipse()
    train_nonlinear_firstorder_ellipse()


if __name__ == "__main__":
    # init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    torch.set_default_dtype(torch.float64)
    main()
