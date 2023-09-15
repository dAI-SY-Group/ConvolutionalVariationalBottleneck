from collections import defaultdict, OrderedDict

import torch
import numpy as np

from src import utils
from src.metrics import get_metric_function
from src.utils import timer, get_dataset_mean_dev_tensors, EarlyStopping, set_optimizer, set_scheduler
from src.loss import TV, get_loss_function
from src.models.modules import MedianPool2d

def get_attack(attack, dataset, model, victim_inputs, victim_targets, config, *args, **kwargs):
    if attack == 'IG':
        return IG(dataset, model, victim_inputs, victim_targets, config, *args, **kwargs)
    elif attack == 'CPL':
        return CPL(dataset, model, victim_inputs, victim_targets, config, *args, **kwargs)
    elif attack == 'iDLG':
        return iDLG(dataset, model, victim_inputs, victim_targets, config, *args, **kwargs)
    else:
        raise NotImplementedError(f'Attack {attack} is not implemented yet!')

class IterativeGradientInversionAttack:
    def __init__(self, dataset, model, victim_inputs, victim_targets, config, *args, **kwargs):
        super().__init__()
        self.config = config.attack
        self.model = model # attacked model
        self.device = config.device
        self.dataset = dataset
        self.data_mean, self.data_std = get_dataset_mean_dev_tensors(self.dataset)
        self.data_mean, self.data_std = self.data_mean.to(self.device), self.data_std.to(self.device)
        if self.config.train_mode:
            self.model.train()
        else:
            self.model.eval()
        self.victim_inputs = victim_inputs
        self.victim_targets = victim_targets
        self.input_shape = victim_inputs[0].shape[1:]

        self.num_classes = config.num_classes
        self.victim_batchsize = self.config.victim_batchsize
        print(f'Victim batchsize:', self.victim_batchsize)
        self.max_trials = max(self.config.max_trials, 1) if self.config.max_trials is not None else 1
        self.max_iterations = self.config.max_iterations
        print(f'Max iterations:', self.max_iterations)
        self.init_method = self.config.init
        self.early_stopper = EarlyStopping(self.config.early_stopping.patience, self.config.early_stopping.delta, self.config.early_stopping.metric, self.config.early_stopping.use_loss, self.config.early_stopping.subject_to, None, self.config.early_stopping.verbose) if self.config.early_stopping else None

        self.metrics = [get_metric_function(m) for m in self.config.metrics]
        self.model_loss_fn = get_loss_function(config.training.loss, self.model)
        self.grad_loss_fn = get_loss_function(self.config.reconstruction_loss, self.model, ignore_layers=self.config.ignore_gradient_layers)

    def init_dummies(self):
        dummy_shape = (self.victim_batchsize, *self.input_shape)
        if self.init_method == 'randn':
            dummy_data = torch.randn(dummy_shape)
        elif self.init_method == 'rand':
            dummy_data = (torch.rand(dummy_shape)-0.5)*2
        else:
            raise NotImplementedError(f'{self.init_method} as dummy data initialization is not implemented yet. Try one of: rand, randn!')
        dummy_label = torch.randn((self.victim_batchsize, self.num_classes)).to(self.device).requires_grad_(True) # TODO see num classes in init.. THIS IS ONLY for classification tasks
        return dummy_data.to(self.device).requires_grad_(True), dummy_label 
    
    def init_attack_specifics(self, vic_target):
        torch.cuda.empty_cache()
        dummy_data, _ = self.init_dummies()
        optimizer = set_optimizer([dummy_data,], self.config.optimizer)
        return dummy_data, vic_target, optimizer

    #attack whole victim data batchwise
    @timer
    def attack(self, batches = None):
        """
        Attack the victim inputs.

        Args:
            batches (int): number of batches to attack. If None, attack all victim inputs.
        """
        batches = len(self.victim_targets) if batches is None else batches
        recons = []
        results = []
        for b, (input_batch, target_batch) in enumerate(zip(self.victim_inputs, self.victim_targets)):
            if input_batch.shape[0] != self.victim_batchsize:
                print(f'Last batch is skipped since it only has a batchsize of {input_batch.shape[0]} instead of {self.victim_batchsize}!')
                continue
            self.current_batch = b
            batch_recon = {}
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            if self.config.train_mode:
                self.model.train()
            else:
                self.model.eval()
            
            gradient, _ = self.get_gradient(input_batch, target_batch)
            
            print(f'Attacking victim batch {b}!')
            reconstructions, rec_labels, best_metrics = self.attack_batch(gradient, target_batch)

            input_batch = input_batch.detach().clone().cpu()
            reconstructions = reconstructions.detach().clone().cpu()

            #reconstructions = self.match_reconstructions(input_batch, reconstructions)

            batch_results = self.rate_reconstruction_quality(input_batch, reconstructions, verbose=True)
            if best_metrics is not None: # None in case of e.g. analytical attacks 
                for metric_name, best in best_metrics.items():
                    batch_results[metric_name] = torch.Tensor([best for _ in range(input_batch.shape[0])]) if self.victim_batchsize > 1 else best
                bl = best_metrics['ReconstructionLoss']
                bi = best_metrics['BestIteration']
                print(f'The reconstruction reached a best reconstruction loss of {bl:.7f} after {bi} iterations.')

            batch_recon['Targets'] = target_batch.detach().clone().cpu()
            batch_recon['Inputs'] = input_batch
            batch_recon['Reconstructions'] = reconstructions
            recons.append(batch_recon)
            results.append(batch_results)

            if b+1 == batches:
                break
        return recons, results
    

    def get_gradient(self, input, target, create_graph=False):
        """
        Get the gradient of the loss function w.r.t. the input.

        Args:
            input (torch.Tensor): input to the model.
            target (torch.Tensor): target labels.
            create_graph (bool): whether to create a graph for backpropagation.
        """
        self.model.zero_grad()
        prediction = self.model(input)
        target_loss = self.model_loss_fn(prediction, target)

        gradient = torch.autograd.grad(target_loss, self.model.parameters(), create_graph=create_graph)

        named_gradients = OrderedDict()
        for grad, (name, param) in zip(gradient, self.model.named_parameters()):
            named_gradients[name] = grad if create_graph else grad.detach()
        return named_gradients, prediction

    @timer
    def attack_batch(self, gradient, vic_target):
        """
        Attack a batch of victim inputs.

        Args:
            gradient (OrderedDict): gradient of the loss function w.r.t. the input.
            vic_target (torch.Tensor): target labels.
        """
        best_trial_metrics = defaultdict(list)
        recon_trials = []
        label_trials = []
        
        for trial in range(self.max_trials): 
            self.current_trial = trial
            dummy_data, dummy_label, optimizer = self.init_attack_specifics(vic_target) 
            scheduler = set_scheduler(optimizer, 1, self.config.scheduler) if self.config.scheduler else None 

            for i in range(self.max_iterations):
                self.current_iteration = i
                closure = self._gradient_closure(optimizer, dummy_data, dummy_label, gradient)
                reconstruction_loss = optimizer.step(closure)

                with torch.no_grad():
                    if self.config.clip_dummies:
                        dummy_data.data = torch.max(torch.min(dummy_data, (1 - self.data_mean) / self.data_std), -self.data_mean / self.data_std)

                    if self.filter and (i+1) % self.config.dummy_filter.iteration == 0:
                        dummy_data.data = self.filter(dummy_data)

                if hasattr(self, 'attack_specific_post_processing'):
                    self.attack_specific_post_processing()

                if scheduler:
                    if scheduler.after_val:
                        scheduler.step(reconstruction_loss)
                    else:
                        scheduler.step()

                # save best trial
                if self.early_stopper:
                    self.early_stopper(reconstruction_loss)
                    if self.early_stopper.improved:
                        best_loss = reconstruction_loss
                        best_iteration = self.current_iteration
                        best_reconstructed_data = dummy_data.detach().clone()
                        best_reconstructed_label = dummy_label.detach().clone()

                    if self.early_stopper.stop:
                        print(f'Early stopping the reconstruction since we had no improvement of {self.early_stopper.metric} for {self.early_stopper.patience} rounds.')
                        break
                else:
                    # if we don't have an early stopper, we just keep the best reconstruction
                    best_loss = reconstruction_loss
                    best_iteration = self.current_iteration
                    best_reconstructed_data = dummy_data.detach()
                    best_reconstructed_label = dummy_label.detach()

                if best_loss < 0.00001:
                    print(f'Early Stopping since the reconstruction loss is extremely low.')
                    break
            
            print(f'Reconstruction was stopped after {self.current_iteration+1} iterations for batch {self.current_batch} trial {self.current_trial}')

            recon_trials.append(best_reconstructed_data)
            label_trials.append(best_reconstructed_label)
 
            best_trial_metrics['ReconstructionLoss'].append(best_loss)
            best_trial_metrics['BestIteration'].append(best_iteration)
            best_trial_metrics['NeededIterations'].append(self.current_iteration)

            best_trial = np.argmin(best_trial_metrics['ReconstructionLoss'])

            if self.early_stopper:
                self.early_stopper.reset()
        
        if self.max_trials > 1:
            print(f'Done with iterative reconstruction of this batch! Out of {self.max_trials} trials trial {best_trial+1} was the best!')

        best_metrics = {k: v[best_trial] for k, v in best_trial_metrics.items()}

        return recon_trials[best_trial], label_trials[best_trial], best_metrics

    def _gradient_closure(self, optimizer, dummy_data, label, gradient):
        """
        Closure for the gradient attack.

        Args:
            optimizer (torch.optim.Optimizer): optimizer for the attack.
            dummy_data (torch.Tensor): dummy data.
            label (torch.Tensor): label.
            gradient (OrderedDict): gradient of the loss function w.r.t. the input.
        """
        def closure(): 
            optimizer.zero_grad()
            if self.config.attacker_train_mode:
                self.model.train()
            else:
                self.model.eval()
            # get the gradient of the loss function w.r.t. the input
            dummy_grad, prediction = self.get_gradient(dummy_data, label, create_graph=True)
            # compute the attack loss
            inversion_loss = self.grad_loss_fn(dummy_grad, gradient)

            # compute the regularization loss
            if hasattr(self, 'attack_specific_regularization'):
                inversion_loss = self.attack_specific_regularization(inversion_loss, dummy_data=dummy_data, gradient=gradient, dummy_gradient=dummy_grad, prediction=prediction, label=label)
                
            inversion_loss.backward() 

            if self.config.use_grad_signs:
                dummy_data.grad.sign_()

            return inversion_loss.item()
        return closure


    def rate_reconstruction_quality(self, inputs, reconstructions, verbose=False):
        results = {}
        for metric_fn in self.metrics:
            vals = []
            for single_input, single_reconstruction in zip(inputs, reconstructions):
                vals.append(metric_fn(single_input.unsqueeze(0), single_reconstruction.unsqueeze(0)).detach().cpu())
            vals = torch.stack(vals)
            results[metric_fn.name] = vals
            if verbose:
                print_val = vals if len(vals.shape)>1 else vals.mean()
                print(f'The reconstruction reached a mean {metric_fn.name} score of {print_val:{metric_fn.format}}.')
        return results



class IG(IterativeGradientInversionAttack):
    def __init__(self, dataset, model, victim_inputs, victim_targets, config, *args, **kwargs):
        super().__init__(dataset, model, victim_inputs, victim_targets, config, *args, **kwargs)
        self.filter = MedianPool2d(kernel_size=self.config.dummy_filter.kernel, stride=self.config.dummy_filter.stride, padding=self.config.dummy_filter.padding, same=False) if self.config.dummy_filter is not None else None

    def attack_specific_regularization(self, inversion_loss, *args, **kwargs):
        if self.config.regularization:
            regularization_loss = self.config.regularization * TV(kwargs['dummy_data'])
            return inversion_loss + regularization_loss
        else: 
            return inversion_loss
        
class CPL(IterativeGradientInversionAttack):
    def __init__(self, model, victim_inputs, victim_targets, config, *args, **kwargs):
        super().__init__(model, victim_inputs, victim_targets, config, *args, **kwargs)
        self.filter = None

    def attack_specific_regularization(self, inversion_loss, *args, **kwargs):
        if self.config.regularization:
            regularization_loss = self.config.regularization * torch.dist(kwargs['prediction'], kwargs['label'], 2)
            return inversion_loss + regularization_loss
        else: 
            return inversion_loss

class iDLG(IterativeGradientInversionAttack):
    def __init__(self, model, victim_inputs, victim_targets, config, *args, **kwargs):
        super().__init__(model, victim_inputs, victim_targets, config, *args, **kwargs)
        self.regularization = None
        self.filter = None
    