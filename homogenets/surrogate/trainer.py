import time
import csv
import torch
import matplotlib.pyplot as plt
from . import neural_net


class ModelTrainer:
    def __init__(
        self,
        model: neural_net.NN,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.logged_epoch = []
        self.logged_L_train = []
        self.logged_E_train = []
        self.logged_L_val = []
        self.logged_E_val = []

    def train(self, model_fname, num_epochs, log_every_X, verbose=True):
        start_time = time.time()
        for i in range(num_epochs):
            train_loss = self.train_one_epoch(i)
            if i % log_every_X == 0:
                val_loss = self.validation_step()
                self.log_epoch(
                    i, time.time() - start_time, train_loss, val_loss, verbose
                )
        # self.plot_convergence()
        self.save(model_fname)

    def train_one_epoch(self, epoch):
        self.model.train(True)
        avg_loss = 0.0
        num_samples = 0
        for i, (x, y) in enumerate(self.train_dataloader):
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            yhat = self.model(x)

            # train
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()

            # Gather data and report
            avg_loss += loss.detach().item()
            num_samples += yhat.shape[0]
        # self.scheduler.step()
        return avg_loss / num_samples

    def validation_step(self):
        self.model.train(False)
        self.optimizer.zero_grad()
        avg_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_dataloader):
                yhat = self.model(x)
                loss = self.loss_fn(yhat, y)

                avg_loss += loss.detach().item()
                num_samples += yhat.shape[0]
        return avg_loss / num_samples

    def eval_step(self):
        # calculate mean average percent error (mape)
        self.model.train(False)
        self.optimizer.zero_grad()
        mape_train = 0.0
        mape_val = 0.0
        with torch.no_grad():
            # train
            num_samples = 0
            for i, (x, y) in enumerate(self.train_dataloader):
                yhat = self.model(x)

                err_y = self.eval_error_fn(yhat, y)
                mape_train += err_y
                num_samples += yhat.shape[0]
            mape_train /= num_samples

            # val
            num_samples = 0
            for i, (x, y) in enumerate(self.val_dataloader):
                yhat = self.model(x)
                err_y = self.eval_error_fn(yhat, y)
                mape_val += err_y
                num_samples += yhat.shape[0]
            mape_val /= num_samples
        return mape_train, mape_val

    def eval_error_fn(self, yhat, y):
        # err_num = 0.0;
        # err_cap = 1.e2
        # rel_error = torch.nan_to_num(
        #        (y - yhat) / y,
        #        nan = err_num, posinf=err_num, neginf=err_num)
        # rel_error[rel_error < -err_cap] = err_num
        # rel_error[rel_error >  err_cap] = err_num
        # rel_error =  torch.sum(torch.abs(rel_error)).item()

        # rel_error = (torch.norm(y-yhat)/torch.norm(y)).item()

        # rel_error = torch.sum((y-yhat)**2)/torch.sum(y**2).item()

        rel_error = torch.sum(
            torch.linalg.vector_norm(y - yhat, dim=1)
            / torch.linalg.vector_norm(y, dim=1)
        ).item()

        # rel_error = torch.sqrt( (torch.mean(torch.square(y-yhat)) /
        #                         torch.sum(torch.square(y))) ).item()

        return rel_error

    def log_epoch(self, i, elapsed, train_loss, val_loss, verbose):
        # evaluate error using an alternative metric
        err_train, err_val = self.eval_step()

        # save progress
        self.logged_epoch.append(i)
        self.logged_L_train.append(train_loss)
        self.logged_E_train.append(err_train)
        self.logged_L_val.append(val_loss)
        self.logged_E_val.append(err_val)

        if verbose:
            print(
                "Epoch # {:04d}, T {:05.0f}s || L_tr {:.3e} L_val {:.3e} || err_tr {:.3e} err_val {:.3e}".format(
                    i, elapsed, train_loss, val_loss, err_train, err_val
                )
            )

    def plot_convergence(self):
        fig = plt.figure(1, figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        lw = 4
        ms = 4
        markers = ["D", "o", "+", "x"]
        colors = "blue", "green", "red", "orange"
        labels = [
            "Training Loss",
            "Validation Loss",
            "Training Error",
            "Validation Error",
        ]
        x = self.logged_epoch
        ys = [
            self.logged_L_train,
            self.logged_L_val,
            self.logged_E_train,
            self.logged_E_val,
        ]

        for y, marker, label, color in zip(ys, markers, labels, colors):
            ax.plot(
                x,
                y,
                linestyle="-",
                marker=marker,
                c=color,
                lw=lw / 2,
                ms=ms,
                label=label,
                markerfacecolor="none",
                markeredgewidth=lw,
            )
        ax.ticklabel_format(style="sci", scilimits=(0, 0))
        plt.legend(loc="upper right")
        plt.show()
        fig.clf()

    def save(self, model_fname):
        # save training progress
        file_opt = "w"
        self.list_to_csv(model_fname + "_epochs", file_opt, self.logged_epoch)
        self.list_to_csv(model_fname + "_loss_tr", file_opt, self.logged_L_train)
        self.list_to_csv(model_fname + "_loss_val", file_opt, self.logged_L_val)
        self.list_to_csv(model_fname + "_err_tr", file_opt, self.logged_E_train)
        self.list_to_csv(model_fname + "_err_val", file_opt, self.logged_E_val)

    def list_to_csv(self, fname, file_opt, datalist):
        with open(fname, file_opt, newline="") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(datalist)


class ModelAndJacobianTrainer:
    def __init__(
        self,
        model: neural_net.NNwithJacobian,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        dim_jac_fwd,  # the jacobian of interest is directly output to these cols
        train_fwd=True,  # train the forward pass
        train_jacobian=True,  # train the accuracy of backpropagation
        jacobian_scale=1,  # divide the jacobian's loss by this value (if 0, then loss's are normalized)
        schedule=False,
        noise=0,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        # self.loss_fn = RL2Loss()
        self.dim_jac_fwd = dim_jac_fwd
        self.train_fwd = train_fwd
        self.train_jacobian = train_jacobian
        self.jacobian_scale = jacobian_scale
        self.schedule = schedule
        self.noise = noise

        self.logged_epoch = []
        self.logged_L_train = []
        self.logged_E_train = []
        self.logged_L_val = []
        self.logged_E_val = []
        self.logged_jac_train = []
        self.logged_jac_val = []
        self.logged_dx_train = []
        self.logged_dx_val = []

    def train(self, model_fname, num_epochs, log_every_X, verbose=True):
        if self.schedule:
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_epochs)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.5
            )
        start_time = time.time()
        for i in range(num_epochs):
            train_loss = self.train_one_epoch(i)
            if i % log_every_X == 0:
                val_loss = self.validation_step()
                self.log_epoch(
                    i, time.time() - start_time, train_loss, val_loss, verbose
                )
        # self.plot_convergence()
        self.save(model_fname)

    def train_one_epoch(self, epoch):
        self.model.train(True)
        avg_loss = 0.0
        num_samples = 0
        for i, (x, y) in enumerate(self.train_dataloader):
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # apply noise to input
            if self.noise != 0.0:
                noise = 1 + self.noise * torch.randn((x.shape))
                x *= noise

            # Make predictions for this batch
            yhat = self.model(x)
            yhat_i, dyhat_dx = self.model.jacobian_from_fwd(yhat)
            jachat = self.model.jacobian_from_bwd(yhat)
            y_i, jac = self.model.jacobian_from_fwd(y, self.dim_jac_fwd)

            # loss
            loss_y = 0.0
            loss_jac = 0.0
            loss_dydx = 0.0
            if self.train_fwd:
                y_i = self.model.loss_proc(y_i)
                loss_y = self.loss_fn(yhat_i, y_i)
            if self.train_jacobian:
                loss_jac = self.loss_fn(jachat, jac)
                # loss_jac = self.custom_loss(jachat, jac)
            if dyhat_dx is not None:
                loss_dydx = self.loss_fn(dyhat_dx, jac)
                # loss_dydx = self.custom_loss(dyhat_dx, jac)

            # train
            loss = loss_y + loss_jac + loss_dydx
            loss.backward()
            self.optimizer.step()

            # Gather data and report
            # avg_loss += (loss_y + loss_jac + loss_dydx).detach().item()
            avg_loss += loss.detach().item()
            num_samples += y.shape[0]
        if self.schedule:
            self.scheduler.step()
        return avg_loss / num_samples

    def validation_step(self):
        self.model.train(False)
        self.optimizer.zero_grad()
        avg_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_dataloader):
                yhat = self.model(x)
                yhat_i, dyhat_dx = self.model.jacobian_from_fwd(yhat)
                jachat = self.model.jacobian_from_bwd(yhat)
                y_i, jac = self.model.jacobian_from_fwd(y, self.dim_jac_fwd)

                # loss
                yhat_i = self.model.de_loss_proc(yhat_i)
                loss_y = self.loss_fn(yhat_i, y_i)
                if self.train_jacobian:
                    loss_jac = self.loss_fn(jachat, jac)
                else:
                    loss_jac = 0.0
                if dyhat_dx is not None:
                    loss_dydx = self.loss_fn(dyhat_dx, jac)
                else:
                    loss_dydx = 0.0
                avg_loss += (loss_y + loss_jac + loss_dydx).detach().item()
                num_samples += yhat.shape[0]
        return avg_loss / num_samples

    def eval_step(self):
        # calculate mean average percent error (mape)
        self.model.train(False)
        self.optimizer.zero_grad()
        mape_train = 0.0
        mape_train_jac = 0.0
        mape_train_dx = 0.0
        mape_val = 0.0
        mape_val_jac = 0.0
        mape_val_dx = 0.0
        num_samples = 0
        with torch.no_grad():
            # train
            for i, (x, y) in enumerate(self.train_dataloader):
                yhat = self.model(x)
                yhat_i, dyhat_dx = self.model.jacobian_from_fwd(yhat)
                jachat = self.model.jacobian_from_bwd(yhat)
                y_i, jac = self.model.jacobian_from_fwd(y, self.dim_jac_fwd)
                yhat_i = self.model.de_loss_proc(yhat_i)

                err_y = self.eval_error_fn(yhat_i, y_i)
                if jac is not None and jachat.shape[1] != 0:
                    err_jac = self.eval_error_fn(jachat, jac)
                else:
                    err_jac = 0.0
                if dyhat_dx is not None:
                    err_dydx = self.eval_error_fn(dyhat_dx, jac)
                else:
                    err_dydx = 0.0
                mape_train += err_y
                mape_train_jac += err_jac
                mape_train_dx += err_dydx
                num_samples += yhat.shape[0]
            mape_train /= num_samples
            mape_train_jac /= num_samples
            mape_train_dx /= num_samples

            # val
            num_samples = 0
            for i, (x, y) in enumerate(self.val_dataloader):
                yhat = self.model(x)
                yhat_i, dyhat_dx = self.model.jacobian_from_fwd(yhat)
                jachat = self.model.jacobian_from_bwd(yhat)
                y_i, jac = self.model.jacobian_from_fwd(y, self.dim_jac_fwd)
                yhat_i = self.model.de_loss_proc(yhat_i)

                err_y = self.eval_error_fn(yhat_i, y_i)
                if jac is not None and jachat.shape[1] != 0:
                    err_jac = self.eval_error_fn(jachat, jac)
                else:
                    err_jac = 0.0
                if dyhat_dx is not None:
                    err_dydx = self.eval_error_fn(dyhat_dx, jac)
                else:
                    err_dydx = 0.0
                mape_val += err_y
                mape_val_jac += err_jac
                mape_val_dx += err_dydx
                num_samples += yhat.shape[0]
            mape_val /= num_samples
            mape_val_jac /= num_samples
            mape_val_dx /= num_samples
        return (
            mape_train,
            mape_train_jac,
            mape_train_dx,
            mape_val,
            mape_val_jac,
            mape_val_dx,
        )

    def eval_error_fn(self, yhat, y):
        # err_num = 0.0;
        # err_cap = 1.e2
        # rel_error = torch.nan_to_num(
        #        (y - yhat) / y,
        #        nan = err_num, posinf=err_num, neginf=err_num)
        # rel_error[rel_error < -err_cap] = err_num
        # rel_error[rel_error >  err_cap] = err_num
        # rel_error =  torch.sum(torch.abs(rel_error)).item()

        # rel_error = (torch.norm(y-yhat)/torch.norm(y)).item()

        # rel_error = torch.sum((y-yhat)**2)/torch.sum(y**2).item()

        rel_error = torch.sum(
            torch.linalg.vector_norm(y - yhat, dim=1)
            / torch.linalg.vector_norm(y, dim=1)
        ).item()

        # rel_error = torch.sqrt( (torch.mean(torch.square(y-yhat)) /
        #                         torch.sum(torch.square(y))) ).item()

        return rel_error

    def log_epoch(self, i, elapsed, train_loss, val_loss, verbose):
        # evaluate error using an alternative metric
        err_train, err_train_jac, err_train_dx, err_val, err_val_jac, err_val_dx = (
            self.eval_step()
        )

        # save progress
        self.logged_epoch.append(i)
        self.logged_L_train.append(train_loss)
        self.logged_E_train.append(err_train)
        self.logged_L_val.append(val_loss)
        self.logged_E_val.append(err_val)
        self.logged_jac_train.append(err_train_jac)
        self.logged_jac_val.append(err_val_jac)
        self.logged_dx_train.append(err_train_dx)
        self.logged_dx_val.append(err_val_dx)

        if verbose:
            print(
                "Epoch # {:04d}, T {:05.0f}s || L_tr {:.3e} L_val {:.3e} || err_tr {:.3e} err_val {:.3e} ||\n\tjac_tr {:.3e} jac_val {:.3e} || dx_tr {:.3e} dx_val {:.3e}".format(
                    i,
                    elapsed,
                    train_loss,
                    val_loss,
                    err_train,
                    err_val,
                    err_train_jac,
                    err_val_jac,
                    err_train_dx,
                    err_val_dx,
                )
            )

    def plot_convergence(self):
        fig = plt.figure(1, figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        lw = 4
        ms = 4
        markers = ["D", "o", "+", "x"]
        colors = "blue", "green", "red", "orange"
        labels = [
            "Training Loss",
            "Validation Loss",
            "Training Error",
            "Validation Error",
        ]
        x = self.logged_epoch
        ys = [
            self.logged_L_train,
            self.logged_L_val,
            self.logged_E_train,
            self.logged_E_val,
        ]

        for y, marker, label, color in zip(ys, markers, labels, colors):
            ax.plot(
                x,
                y,
                linestyle="-",
                marker=marker,
                c=color,
                lw=lw / 2,
                ms=ms,
                label=label,
                markerfacecolor="none",
                markeredgewidth=lw,
            )
        ax.ticklabel_format(style="sci", scilimits=(0, 0))
        plt.legend(loc="upper right")
        plt.show()
        fig.clf()

    def save(self, model_fname):
        # save training progress
        file_opt = "w"
        self.list_to_csv(model_fname + "_epochs", file_opt, self.logged_epoch)
        self.list_to_csv(model_fname + "_loss_tr", file_opt, self.logged_L_train)
        self.list_to_csv(model_fname + "_loss_val", file_opt, self.logged_L_val)
        self.list_to_csv(model_fname + "_err_tr", file_opt, self.logged_E_train)
        self.list_to_csv(model_fname + "_err_val", file_opt, self.logged_E_val)
        self.list_to_csv(model_fname + "_err_jac_tr", file_opt, self.logged_jac_train)
        self.list_to_csv(model_fname + "_err_jac_val", file_opt, self.logged_jac_val)
        self.list_to_csv(model_fname + "_err_dx_tr", file_opt, self.logged_dx_train)
        self.list_to_csv(model_fname + "_err_dx_val", file_opt, self.logged_dx_val)

    def list_to_csv(self, fname, file_opt, datalist):
        with open(fname, file_opt, newline="") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(datalist)
