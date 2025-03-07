# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from .hook import Hook

import torch.distributed as dist

class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class ParamUpdateHook(Hook):
    """
    Parameter Update Hook

    necessary for update the model parameters
    """
    
    def before_train_step(self, algorithm):
        if hasattr(algorithm, 'start_run'):
            torch.cuda.synchronize()
            algorithm.start_run.record()

    # call after each train_step to update parameters
    def after_train_step(self, algorithm):
        loss = algorithm.out_dict['loss']
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss).backward()
            if hasattr(algorithm, "prototypes"):
                with torch.no_grad():
                    algorithm.prototypes.grad.data = AllReduceSum.apply(algorithm.prototypes.grad.data)
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler.unscale_(algorithm.optimizer)
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler.step(algorithm.optimizer)
            algorithm.loss_scaler.update()
        else:
            loss.backward()
            if hasattr(algorithm, "prototypes"):
                with torch.no_grad():
                    algorithm.prototypes.grad.data = AllReduceSum.apply(algorithm.prototypes.grad.data)
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.optimizer.step()

        if algorithm.scheduler is not None:
            algorithm.scheduler.step()
        algorithm.model.zero_grad()

        if hasattr(algorithm, 'end_run'):
            algorithm.end_run.record()
            torch.cuda.synchronize()
            algorithm.log_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.

