import math
import torch
from typing import List, Optional
from torch import Tensor
from . import activations


class NN(torch.nn.Module):
    def __init__(
        self,
        seed,
        input_size,
        output_size,
        batch_size,
        hidden_size,
        num_hidden_layers,
        activation_func=None,
        pre_processor=None,
        post_processor=None,
        pre_processor_args=None,
        post_processor_args=None,
        loss_processor=None,
        de_loss_processor=None,
        loss_processor_args=None,
        de_loss_processor_args=None,
    ):
        super(NN, self).__init__()

        # Set defaults
        if activation_func is None:
            activation_func = activations.IdentityActivation
        if pre_processor is None:
            pre_processor = activations.IdentityActivation
        if post_processor is None:
            post_processor = activations.IdentityActivation
        if loss_processor is None:
            loss_processor = activations.IdentityActivation
        if de_loss_processor is None:
            de_loss_processor = activations.IdentityActivation

        # setup model info
        torch.manual_seed(seed)
        self.seed = seed
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # processor
        if pre_processor_args is not None:
            pre_proc = pre_processor(batch_size, hidden_size, *pre_processor_args)
        else:
            pre_proc = pre_processor(batch_size, hidden_size)

        if post_processor_args is not None:
            post_proc = post_processor(batch_size, hidden_size, *post_processor_args)
        else:
            post_proc = post_processor(batch_size, hidden_size)

        if loss_processor_args is not None:
            loss_proc = loss_processor(batch_size, output_size, *loss_processor_args)
            de_loss_proc = de_loss_processor(
                batch_size, output_size, *de_loss_processor_args
            )
        else:
            loss_proc = loss_processor(batch_size, output_size)
            de_loss_proc = de_loss_processor(batch_size, output_size)
        self.loss_proc = loss_proc
        self.de_loss_proc = de_loss_proc

        # build network layers
        self.layers = torch.nn.Sequential()

        # first layer in
        self.layers.append(pre_proc)
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        self.layers.append(activation_func(batch_size, hidden_size))

        # hidden layers
        for _ in range(num_hidden_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation_func(batch_size, hidden_size))

        # last layer out
        self.layers.append(torch.nn.Linear(hidden_size, output_size))
        self.layers.append(post_proc)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                pass  # NOTE
                # torch.nn.init.normal_(m.weight)
                # torch.nn.init.normal_(m.bias)
                ##m.bias.data.fill_(0.)
                # torch.nn.init.uniform_(m.weight)
                # torch.nn.init.uniform_(m.bias)
                # torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.xavier_uniform_(m.bias)
                # torch.nn.init.xavier_normal_(m.weight)
                # torch.nn.init.xavier_normal_(m.bias)

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

    @torch.jit.export
    def backward(self, y):
        B, D = y.shape
        dy_dx = torch.stack(B * [torch.eye(D)])

        # call backwards on the post processing layer
        dy_dx = self.layers[-1].backward(dy_dx)

        # call backwards on the last layer of weights
        dy_dx = dy_dx @ self.layers[-2].weight

        # call backwards on the subsequent layers (hidden -> first)
        backward_layers = self.layers[1:-2][::-1]
        for layer in backward_layers:
            dy_dx = dy_dx @ layer.weight

        # call backwards on the preprocessing layer
        dy_dx = self.layers[0].backward(dy_dx)

        return dy_dx

    @torch.jit.export
    def jacobian(self, x_i):
        """
        https://pytorch.org/functorch/1.13/notebooks/jacobians_hessians.html
        @param x_i the input vector = {xi0, xi1, ... xiD} for D features
        @return [dy0 / dxi0, ... dy0 / dxiD] for each batch
                [            ...           ]
                [dyN / dxi0, ... dyN / dxiD]
        """
        jacobian_rows: List[Tensor] = []
        xp = x_i.clone().requires_grad_()
        unit_vectors = torch.eye(self.output_size)
        for vec in unit_vectors:
            # optional arguments must be cast to Optional
            vec_jit: Optional[Tensor] = torch.stack(
                [vec] * x_i.shape[0], 0
            )  # .reshape((1, self.output_size))
            dy_dx = torch.autograd.grad(
                [
                    self.forward(xp),
                ],
                [
                    xp,
                ],
                grad_outputs=[
                    vec_jit,
                ],
            )[0]
            # Recast the result to be non-optional
            assert dy_dx is not None
            jacobian_rows.append(dy_dx)
        jac = torch.stack(jacobian_rows, 1)
        return jac

    @torch.jit.export
    def jacobian_by_FD(self, x_i, FD_delta: float):
        with torch.no_grad():
            B = x_i.shape[0]
            D = x_i.shape[1]
            y0 = self.forward(x_i.clone())
            N = y0.shape[1]
            jac = torch.zeros(B, N, D)
            for b in range(B):
                for i in range(D):
                    # perturb x
                    x_pert = torch.zeros((B, D))
                    x_pert[b, i] += FD_delta
                    xupp = x_i.clone() + x_pert
                    xlow = x_i.clone() - 2 * x_pert

                    # estimate the derivative
                    with torch.no_grad():
                        yupp = self.forward(xupp.clone())
                        ylow = self.forward(xlow.clone())
                    dy_dx_i = (yupp - ylow) / FD_delta / 3

                    # set the jacobian
                    jac[b, :, i] = dy_dx_i[b]
            return jac


class NNwithJacobian(torch.nn.Module):
    def __init__(
        self,
        seed,
        input_size,
        output_size,
        batch_size,
        hidden_size,
        num_hidden_layers,
        dim_fwd=None,
        dim_jac_in=None,  # the jacobian of interest corresponds to these input cols
        dim_jac_out=None,  # the jacobian of interest corresponds to these output cols
        dim_jac_fwd=None,  # the jacobian of interest is directly output to these cols
        activation_func=None,
        pre_processor=None,
        post_processor=None,
        pre_processor_args=None,
        post_processor_args=None,
        loss_processor=None,
        de_loss_processor=None,
        loss_processor_args=None,
        de_loss_processor_args=None,
        batch_norm=False,
        layer_norm=False,
        dropout=False,
    ):
        super(NNwithJacobian, self).__init__()

        # Set defaults
        if activation_func is None:
            activation_func = activations.IdentityActivation
        if pre_processor is None:
            pre_processor = activations.IdentityActivation
        if post_processor is None:
            post_processor = activations.IdentityActivation
        if loss_processor is None:
            loss_processor = activations.IdentityActivation
        if de_loss_processor is None:
            de_loss_processor = activations.IdentityActivation

        # setup model info
        torch.manual_seed(seed)
        self.seed = seed
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout = dropout

        # processor
        if pre_processor_args is not None:
            pre_proc = pre_processor(batch_size, hidden_size, *pre_processor_args)
        else:
            pre_proc = pre_processor(batch_size, hidden_size)

        if post_processor_args is not None:
            post_proc = post_processor(batch_size, hidden_size, *post_processor_args)
        else:
            post_proc = post_processor(batch_size, hidden_size)

        if loss_processor_args is not None:
            loss_proc = loss_processor(batch_size, output_size, *loss_processor_args)
            de_loss_proc = de_loss_processor(
                batch_size, output_size, *de_loss_processor_args
            )
        else:
            loss_proc = loss_processor(batch_size, output_size)
            de_loss_proc = de_loss_processor(batch_size, output_size)
        self.loss_proc = loss_proc
        self.de_loss_proc = de_loss_proc

        # input cols relative to global dastset
        if dim_fwd is None:
            self.dim_fwd = (0, input_size)
        else:
            self.dim_fwd = dim_fwd

        # jacobian inputs
        if dim_jac_in is None:
            self.dim_jac_in = (0, input_size)
        else:
            self.dim_jac_in = dim_jac_in
        if dim_jac_out is None:
            self.dim_jac_out = (0, output_size)
        else:
            self.dim_jac_out = dim_jac_out
        self.dim_jac_fwd = dim_jac_fwd

        # build network layers
        self.layers = torch.nn.Sequential()

        # first layer in
        self.layers.append(pre_proc)
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        self.layers.append(activation_func(batch_size, hidden_size))

        # hidden layers
        for _ in range(num_hidden_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation_func(batch_size, hidden_size))

        # last layer out
        self.layers.append(torch.nn.Linear(hidden_size, output_size))
        self.layers.append(post_proc)

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                pass  # NOTE
                # torch.nn.init.normal_(m.weight)
                # torch.nn.init.normal_(m.bias)
                ##m.bias.data.fill_(0.)
                # torch.nn.init.uniform_(m.weight)
                # torch.nn.init.uniform_(m.bias)
                # torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.xavier_uniform_(m.bias)
                # torch.nn.init.xavier_normal_(m.weight)
                # torch.nn.init.xavier_normal_(m.bias)

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

    @torch.jit.export
    def backward(self, y):
        B, D = y.shape
        dy_dx = torch.stack(B * [torch.eye(D)])

        # call backwards on the post processing layer
        dy_dx = self.layers[-1].backward(dy_dx)

        # call backwards on the last layer of weights
        dy_dx = dy_dx @ self.layers[-2].weight

        # call backwards on the subsequent layers (hidden -> first)
        backward_layers = self.layers[1:-2][::-1]
        for layer in backward_layers:
            dy_dx = dy_dx @ layer.weight

        # call backwards on the preprocessing layer
        dy_dx = self.layers[0].backward(dy_dx)

        return dy_dx

    def jacobian_from_fwd(self, yhat, dim_jac_fwd=None):
        # use the provided dim_jac_fwd, or fall back to the model's
        if dim_jac_fwd is None:
            if self.dim_jac_fwd is None:
                return yhat, None
            else:
                dim_jac_fwd = self.dim_jac_fwd

        y_i = yhat[:, self.dim_fwd[0] : self.dim_fwd[1]]
        dyhat_dx = yhat[:, dim_jac_fwd[0] : dim_jac_fwd[1]]

        # get the jacobian matrix in style B, N, D
        jac_dim = math.isqrt(dim_jac_fwd[1] - dim_jac_fwd[0])
        jacobian = torch.zeros((yhat.shape[0], jac_dim, jac_dim))
        idx = 0
        for i in range(jac_dim):
            for j in range(jac_dim):
                jacobian[:, i, j] = dyhat_dx[:, idx]
                idx += 1

        # flatten B, N*D
        jacobian = jacobian.reshape((jacobian.shape[0], -1))
        if jacobian.shape[1] == 0:
            jacobian = None

        return y_i, jacobian

    def jacobian_from_bwd(self, yhat):
        dyhatdx = self.backward(yhat)  # B, N, D
        dyhatdx = dyhatdx[
            :,
            self.dim_jac_out[0] : self.dim_jac_out[1],
            self.dim_jac_in[0] : self.dim_jac_in[1],
        ]

        # flatten B, N*D
        dyhatdx = dyhatdx.reshape((dyhatdx.shape[0], -1))
        return dyhatdx

    @torch.jit.export
    def jacobian(self, x_i):
        """
        https://pytorch.org/functorch/1.13/notebooks/jacobians_hessians.html
        @param x_i the input vector = {xi0, xi1, ... xiD} for D features
        @return [dy0 / dxi0, ... dy0 / dxiD] for each batch
                [            ...           ]
                [dyN / dxi0, ... dyN / dxiD]
        """
        jacobian_rows: List[Tensor] = []
        xp = x_i.clone().requires_grad_()
        unit_vectors = torch.eye(self.output_size)
        for vec in unit_vectors:
            # optional arguments must be cast to Optional
            vec_jit: Optional[Tensor] = torch.stack(
                [vec] * x_i.shape[0], 0
            )  # .reshape((1, self.output_size))
            dy_dx = torch.autograd.grad(
                [
                    self.forward(xp),
                ],
                [
                    xp,
                ],
                grad_outputs=[
                    vec_jit,
                ],
            )[0]
            # Recast the result to be non-optional
            assert dy_dx is not None
            jacobian_rows.append(dy_dx)
        jac = torch.stack(jacobian_rows, 1)
        return jac

    @torch.jit.export
    def jacobian_by_FD(self, x_i, FD_delta: float):
        with torch.no_grad():
            B = x_i.shape[0]
            D = x_i.shape[1]
            y0 = self.forward(x_i.clone())
            N = y0.shape[1]
            jac = torch.zeros(B, N, D)
            for b in range(B):
                for i in range(D):
                    # perturb x
                    x_pert = torch.zeros((B, D))
                    x_pert[b, i] += FD_delta
                    xupp = x_i.clone() + x_pert
                    xlow = x_i.clone() - 2 * x_pert

                    # estimate the derivative
                    with torch.no_grad():
                        yupp = self.forward(xupp.clone())
                        ylow = self.forward(xlow.clone())
                    dy_dx_i = (yupp - ylow) / FD_delta / 3

                    # set the jacobian
                    jac[b, :, i] = dy_dx_i[b]
            return jac

    @torch.jit.export
    def jacobian_of_bwd(self, y_i):
        """
        https://pytorch.org/functorch/1.13/notebooks/jacobians_hessians.html
        @param y_i the output vector = {yi0, yi1, ... yiN} for N output
        @return [dy0 / dxi0, ... dy0 / dxiD] for each batch
                [            ...           ]
                [dyN / dxi0, ... dyN / dxiD]
        """
        jacobian_rows: List[Tensor] = []
        xp = y_i.clone().requires_grad_()
        unit_vectors = torch.eye(self.output_size)
        for vec in unit_vectors:
            # optional arguments must be cast to Optional
            vec_jit: Optional[Tensor] = torch.stack(
                [vec] * y_i.shape[0], 0
            )  # .reshape((1, self.output_size))
            dy_dx = torch.autograd.grad(
                [
                    self.backward(xp),
                ],
                [
                    xp,
                ],
                grad_outputs=[
                    vec_jit,
                ],
            )[0]
            # Recast the result to be non-optional
            assert dy_dx is not None
            jacobian_rows.append(dy_dx)
        jac = torch.stack(jacobian_rows, 1)
        return jac
