import os
import copy
from collections import defaultdict
import gc

import numpy as np
import dill
import torch

from src.utils import timer, set_optimizer, set_scheduler, EarlyStopping
from src.loss import get_loss_function
from src.metrics import get_metric_function
from src.history import History

class FederatedTrainer:
    def __init__(self, model, config, device, *args, **kwargs):
        print(f'# START Initialization of FederatedTrainer!')
        assert config is not None, 'A config munch is necessary to initialize the trainer, but config is None!'
        self.model = model
        self.trn_data, self.tst_data, self.val_data = None, None, None
        self.config = config
        self.device = device
        self.debug = config.debug
        self.debug_file_suffix = '_debug' if self.debug else ''

        self.use_val = self.config.data.use_val

        self.loss_fn = get_loss_function(config.training.loss, model=self.model)
        eval_ds_name = self.loss_fn.name + '_L'
        self.loss_fn.eval_ds_name = 'VAL_' + eval_ds_name if self.use_val else 'TST_' + eval_ds_name
        
        self.metrics = []
        for m in config.training.metrics:
            self.metrics.append(get_metric_function(m, num_classes = self.config.num_classes, model=self.model).to(self.device))

        self._set_optimizers()

        self.current_epoch = 0
        
        self.history = History(keys=['Metric', 'Epoch'], savefile=os.path.join(config.history_path, config.experiment_name+self.debug_file_suffix))
        self.checkpoint_path = os.path.join(config.checkpoint_path, config.experiment_name)

        self.overall_trained_epochs = 0
        if config.training.num_clients > 20:
            self.muted = True
        else:
            self.muted = False
        print(f'# END Initialization of FederatedTrainer!')

    def _set_optimizers(self):
        self.optimizer = set_optimizer(self.model, self.config.training.optimizer)
        if self.config.training.lr_scheduler:
            self.scheduler = set_scheduler(self.optimizer, self.config.training.validation_frequency, self.config.training.lr_scheduler, self.use_val)
        else:
            self.scheduler = None
        if self.config.training.early_stopping:
            self.early_stopper = EarlyStopping(self.config.training.early_stopping.patience, self.config.training.early_stopping.delta, self.config.training.early_stopping.metric, self.config.training.early_stopping.use_loss, self.config.training.early_stopping.subject_to, self.use_val, self.config.training.early_stopping.verbose) 
        else: 
            self.early_stopper = None

    def track_metrics(self, outputs, targets):
        for metric_fn in self.metrics:
            metric_fn(outputs, targets)

    def compute_metrics(self):
        metrics_dict = {}
        for metric_fn in self.metrics:
            metrics_dict[metric_fn.name] = metric_fn.compute().item()
            metric_fn.reset()
        return metrics_dict

    def update_history(self, metrics_dict, split): 
        for k, v in metrics_dict.items():
            if k in ['OverallTrainedEpochs']:
                metric_name = k
            else:
                metric_name = f'{split}_{k}'
            self.history((metric_name, self.current_epoch), v)

    def log_status(self):
        if self.muted:
            return
        format = self.loss_fn.format
        trn_loss = self.history.get(f'TRN_{self.loss_fn.name}_L')
        log_str = f'Epoch: {self.current_epoch}| LR: {self.optimizer.lr()} | '+ f'Train loss ({self.loss_fn.name}): {trn_loss:{format}}'
        if self.tst_data is not None:
            try:
                tst_loss = self.history.get(f'TST_{self.loss_fn.name}_L')
                log_str += f' | Test loss ({self.loss_fn.name}): {tst_loss:{format}}'
            except: 
                print(f'No test loss available to report!')
        if self.use_val and self.val_data is not None:
            try:
                val_loss = self.history.get(f'VAL_{self.loss_fn.name}_L')
                log_str += f' | Val loss ({self.loss_fn.name}): {val_loss:{format}}'
            except: 
                print(f'No validation loss available to report!')
        print(log_str)

    def get_state_path(self, path, best=False):
        assert self.checkpoint_path or path, 'If the History object has no default checkpoint_path you have to provide a path!'
        path = self.checkpoint_path if path is None else path
        if best:
            path += '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        return statepath

    def summary(self):
        self.history.summary(mode='latest')
        key_value = f'VAL_{self.loss_fn.name}_L' if self.use_val else f'TST_{self.loss_fn.name}_L'
        self.history.summary(key_value=key_value, max_key=self.loss_fn.subject_to=='max', step_key='Epoch', mode='best')

    def train(self, epochs=None, *args, **kwargs):
        self.history = History(keys=['Metric', 'LocalEpoch'])
        self.best_model = None

        if epochs is None:
            epochs = self.config.training.epochs
        assert type(epochs) == int, f'epochs need to be an integer. Got {epochs} ({type(epochs)}) instead!'
        for epoch in range(epochs):
            self.current_epoch += 1
            self.train_step(*args, **kwargs)

            self._pre_val_scheduler_step()

            if epoch % self.config.training.validation_frequency == 0 or epoch == (epochs-1):
                self.evaluate('VAL', *args, **kwargs)
                self.evaluate('TST', *args, **kwargs)

                if self.early_stopper:
                    self.early_stopper(self.history.get(self.early_stopper.metric))

                self._post_val_scheduler_step()

                #self.log_status()
                self.save()

            if self.current_epoch in self.config.training.save_rounds or epoch == (epochs-1):
                self.save(self.checkpoint_path+f'_e{self.current_epoch}')
            
            if self.early_stopper:
                if self.early_stopper.improved:
                    self.save(self.checkpoint_path, best=True)
                if self.early_stopper.stop:
                    print(f'Early stopping the training since we had no improvement of {self.early_stopper.metric} for {self.early_stopper.patience} rounds. Training was stopped after {epoch} epochs')
                    break
            
            if self.config.debug:
                break
            
        #self.save('save_best')

    def _pre_val_scheduler_step(self):
        if self.scheduler and not self.scheduler.after_val:
            self.scheduler.step()
            self.history(('LR', self.current_epoch), self.scheduler.lr())
    
    def _post_val_scheduler_step(self):
        if self.scheduler and self.scheduler.after_val:
            if hasattr(self.scheduler, 'metric'):
                self.scheduler.step(self.history.get(self.scheduler.metric))
            else:
                self.scheduler.step()
            self.history(('LR', self.current_epoch), self.scheduler.lr())

    def train_step(self, *args, **kwargs):
        self.model.to(self.device)
        self.model.train()
        
        losses = []
        for batch_num, (inputs, targets) in enumerate(self.trn_data):
            # zero out grads
            self._zero_grad()

            # Transfer to GPU
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            #forward pass
            outputs = self._model_forward(inputs)

            # Get loss
            loss = self._calculate_loss(outputs, targets)           

            if torch.isnan(loss).any():
                print('loss is nan in batch {}!'.format(batch_num))
                raise Exception('loss is nan in batch {}!'.format(batch_num))

            loss.backward()

            if self.config.training.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clipping)
            
            self._optimizer_step()

            #calculate all metrics
            self.track_metrics(outputs, targets)
            losses.append(loss.item())
            
            if self.config.debug:
                break

        epoch_metrics = self.compute_metrics()
        epoch_metrics[self.loss_fn.name+'_L'] = np.mean(losses)
        self.update_history(epoch_metrics, 'TRN')
        self._zero_grad()

        self.overall_trained_epochs += 1
        self.update_history({'OverallTrainedEpochs': self.overall_trained_epochs}, 'TRN')
        torch.cuda.empty_cache()

    def _zero_grad(self):
        self.optimizer.zero_grad()
        self.model.zero_grad()

    def _model_forward(self, inputs):
        return self.model(inputs)
    
    def _calculate_loss(self, outputs, targets):
        return self.loss_fn(outputs, targets)

    def _optimizer_step(self):
        self.optimizer.step()

    #adjust save for local federated training, as we do not want to save every single client model
    def save(self, path=None, best=False):
        if path is None:
            return
        if best:
            self.best_model = copy.deepcopy(self.model.cpu().state_dict())        
        if path == 'save_best' and self.config.training.save_client_models:
            self.model.load_state_dict(self.best_model)
            self.model.save(self.checkpoint_path+f'_best')
        
    def load(self, path=None, best=False):
        assert self.checkpoint_path or path, 'If the History object has no default checkpoint_path you have to provide a path!'
        path = self.checkpoint_path if path is None else path
        if best:
            path += '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        with open(statepath, 'rb') as file:
            state = dill.load(file)
        self.history.load()
        self.model.load(path)
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        if self.early_stopper is not None:
            self.early_stopper.set_state(state['early_stopper'])
        self.current_epoch = state['current_epoch']

    def load_best(self):
        self.load(self.checkpoint_path, best=True)

    def evaluate(self, split='VAL', *args, **kwargs):
        if 'reset_history' in kwargs.keys() and kwargs['reset_history']:
            self.history = History(keys=['Metric', 'LocalEpoch'])

        if split == 'TRN':
            data = self.trn_data
        elif split == 'TST':
            data = self.tst_data
        elif split == 'VAL':
            data = self.val_data
        else:
            raise ValueError(f'Cannot interpret split "{split}"! Use one of [TRN, TST, VAL]')
        if data is None:
            return
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_num, (inputs, targets) in enumerate(data):
                # Transfer to GPU
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Get loss and metric
                outputs = self._model_forward(inputs)
                losses.append(self._calculate_loss(outputs, targets).item())
                self.track_metrics(outputs, targets)
                if self.config.debug:
                    break
        
        if 'metric_tag' in kwargs:
            split = f'{split}_{kwargs["metric_tag"]}'
        else:
            split = split
        eval_metrics = self.compute_metrics()
        eval_metrics[self.loss_fn.name+'_L'] = np.mean(losses)
        self.update_history(eval_metrics, split)
        
    def _model_forward(self, inputs):
        return self.model(inputs)

    def reset_optimizer_state(self, _all=False, full_reinitialization=False, optimizer_config=None, *args, **kwargs):
        config = self.config.training if optimizer_config is None else optimizer_config
        self.current_epoch = 0
        if config.optimizer.name in ['SGD'] or full_reinitialization:
            self.optimizer = set_optimizer(self.model, config.optimizer) #MASSIVELY LEAKS GPU MEMORY FOR STATEFUL OPTIMIZERS LIKE ADAM
        else:
            self.optimizer.state = defaultdict(dict) # RESET OPTIMIZER STATE without recreating whole optimizer which might lead to memory leaks
        self.scheduler = set_scheduler(self.optimizer, config.validation_frequency, config.lr_scheduler, self.use_val)
        if config.early_stopping:
            self.early_stopper = EarlyStopping(config.early_stopping.patience, config.early_stopping.delta, config.early_stopping.metric, config.early_stopping.use_loss, config.early_stopping.subject_to, self.use_val, config.early_stopping.verbose) if config.early_stopping is not None else None
        else:
            self.early_stopper = None
        if _all:
            self.overall_trained_epochs = 0
        gc.collect()
    
    def get_model_state(self):
        return self.model.cpu().state_dict()

    def get_state(self):
        return {'model': self.model.cpu().state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'scheduler':self.scheduler.state_dict() if self.scheduler else None,
                'early_stopper':self.early_stopper.get_state() if self.early_stopper else None,
                'overall_trained_epochs': self.overall_trained_epochs,
                'model_checkpoint_path': self.checkpoint_path,}


    def set_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def set_state(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler and state_dict['scheduler']:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        if self.early_stopper and state_dict['early_stopper']:
            self.early_stopper.set_state(state_dict['early_stopper'])
        self.overall_trained_epochs = state_dict['overall_trained_epochs']
        self.checkpoint_path = state_dict['model_checkpoint_path']
        gc.collect()

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                'early_stopper': self.early_stopper.get_state() if self.early_stopper is not None else None,
                'current_epoch': self.current_epoch,}
    



class Client:
    def __init__(self, client_id, trn_data, tst_data, val_data, model_trainer):
        """
        client_id: Expects client_id / identifier string (unique)
        trn/tst/val_data: Dataloder (pref. torch) for training, test, and validation data.
        """
        self.config = model_trainer.config
        self.wandb = self.config.wandb
        self.client_id = client_id
        self.trn_data = trn_data
        self.tst_data = tst_data
        self.val_data = val_data
        self.trn_samples = len(trn_data.dataset)
        self.tst_samples = 0 if tst_data == None else len(tst_data.dataset)
        self.val_samples = 0 if val_data == None else len(val_data.dataset)
        self.model_trainer = model_trainer
        if self.config.training.random_client_start:
            self.model_trainer.model.reset_parameters()
        model_trainer.reset_optimizer_state(_all=True)
        self.local_state = copy.deepcopy(model_trainer.get_state())
        self.file_identifier = f'{self.config.experiment_name}_CLIENT{self.client_id}'+self.model_trainer.debug_file_suffix
        self.local_history = History(keys=['Metric', 'LocalEpoch', 'CommunicationRound'], savefile=os.path.join(self.config.history_path, self.file_identifier))
        self.checkpoint_path = os.path.join(self.config.checkpoint_path, self.file_identifier)
        self.local_state['model_checkpoint_path'] = self.checkpoint_path
        self.sent_models = 0
        self.received_models = 0
        print(f'# Client {client_id} initialized. local train samples: {self.trn_samples} | local test samples: {self.tst_samples} | local val samples: {self.val_samples} #')

    def get_sample_number(self, split, get_truth=False):
        """returns the number of samples in a given split of a dataset

        Args:
            split ([str]): one of trn, tst, val
            get_truth (bool, optional): returns the actual number of samples if True (else val_samples may be modified if no val_data was provided to the client). Defaults to False.

        Returns:
            [int]: number of samples
        """
        if split == 'TRN':
            return self.trn_samples
        elif split == 'TST':
            return self.tst_samples
        elif split == 'VAL':
            if get_truth:
                return self.val_samples
            if self.val_samples == 0: print('The Client was not provided with explicit validation data. Using testdata instead.')
            return self.val_samples if self.val_samples > 0 else self.tst_samples
        else:
            raise ValueError(('This data split is not defined for Clients. Choose one of trn, tst, val'))


    def train(self, w_global, communication_round):
        self.set_training_model_state(w_global, communication_round)
    
        self.model_trainer.set_state(self.local_state)
        self.model_trainer.reset_optimizer_state()
        self.set_trainer_data()
        self.model_trainer.train()
        self.model_trainer.history.add_col('CommunicationRound', communication_round)
        self.local_history.update(self.model_trainer.history)#, f'CLIENT_{self.client_id}')
        self.local_history(('SentModels', self.model_trainer.current_epoch, communication_round), self.sent_models)
        self.local_history(('ReceivedModels', self.model_trainer.current_epoch, communication_round), self.received_models)
        self.local_state = copy.deepcopy(self.model_trainer.get_state())
        return self._train_output()

    def _train_output(self):
        return self.local_state['model']

    def set_training_model_state(self, w_global, communication_round):
        self.local_state['model'] = w_global # whole model is exchanged!

    def evaluate(self, communication_round, split='TST', model_state=None, *args, **kwargs):
        self.set_trainer_data()

        if model_state is None: #Local training evaluation
            self.model_trainer.evaluate(split, reset_history=True)
            self.model_trainer.history.add_col('CommunicationRound', communication_round)
            self.local_history.update(self.model_trainer.history)
            return None
        else: #an explicit model was given i.e. global aggregated model is to be evaluated
            self.model_trainer.set_model_state(model_state)
            self.model_trainer.evaluate(split, reset_history=True)
            self.model_trainer.history.add_col('CommunicationRound', communication_round)
            self.local_history.update(self.model_trainer.history)
            return self.model_trainer.history

    def set_trainer_data(self):
        self.model_trainer.trn_data = self.trn_data
        self.model_trainer.tst_data = self.tst_data
        self.model_trainer.val_data = self.val_data


    def summary(self):
        print(f'### Local History Summary for Client {self.client_id} ###')
        try:
            #self.local_history.summary(step_key='CommunicationRound', mode='latest')
            key_value = 'VAL_' + self.model_trainer.loss_fn.name  if self.config.data.use_val else 'TST_' + self.model_trainer.loss_fn.name
            key_value += '_L'
            self.local_history.summary(key_value=key_value, max_key=self.model_trainer.loss_fn.subject_to=='max', step_key='CommunicationRound', mode='best')
        except Exception as e:
            print(f'Could not print local history summary for client {self.client_id}! Error: {e}')
            #print('No history available for this client! This client has not participated in training.')     


class SiloFederatedAveragingEnvironment:
    def __init__(self, fed_dataset, model_trainer, config, *args, **kwargs):
        assert config.training.glob is not None, f'You need to provide config.training.glob for a FederatedEnvironment! Got {config.training.glob}'
        print('### Initializing FederatedEnvironment (START) ###')
        self.config = config
        self.debug = self.config.debug
        self.model_trainer = model_trainer
        self._setup_clients(fed_dataset, model_trainer)
        self.model_trainer.model.reset_parameters()
        print('Initializing global model.')
        self.global_model = self.model_trainer.get_model_state()
        self.total_num_exchanged_models = 0
        self.scheduler = None

        
        if self.config.training.glob.early_stopping:
            print('Initializing global EarlyStopper.')
            self.early_stopper = EarlyStopping(config.training.glob.early_stopping.patience, config.training.glob.early_stopping.delta, config.training.glob.early_stopping.metric, config.training.glob.early_stopping.use_loss, config.training.glob.early_stopping.subject_to, self.config.data.use_val, config.training.glob.early_stopping.verbose) if config.training.glob.early_stopping is not None else None
        else:
            self.early_stopper = None

        self.metrics = model_trainer.metrics
        self.loss = model_trainer.loss_fn

        self.file_identifier = f'{self.config.experiment_name}_GLOBAL'+self.model_trainer.debug_file_suffix
        self.global_history = History(keys=['Metric', 'CommunicationRound'], savefile=os.path.join(self.config.history_path, self.file_identifier))
        self.checkpoint_path = os.path.join(self.config.checkpoint_path, self.file_identifier)

        self.current_communication_round = 0

    def _setup_clients(self, fed_dataset, model_trainer, *args, **kwargs):
        print('# Setting up Clients (START) #')
        self.clients = {}
        self.client_id_map = {}
        local_trn_data_dict, local_tst_data_dict, local_val_data_dict = fed_dataset
        for numerical_id, client_id in enumerate(local_trn_data_dict.keys()):
            self.clients[client_id] = Client(client_id, local_trn_data_dict[client_id], local_tst_data_dict[client_id], local_val_data_dict[client_id], model_trainer)
            self.client_id_map[numerical_id] = client_id
        self.num_clients = len(self.clients)
        print(f'# A total of {self.num_clients} clients were initialized. #')
        print('# Setting up Clients (END) #')


    @timer
    def train(self, communication_rounds=None, *args, **kwargs):
        print(f'### {self.__class__.__name__} Training (START) ###')
        if communication_rounds is None:
            communication_rounds = self.config.training.glob.communication_rounds
        for global_epoch in range(communication_rounds):
            self.current_communication_round += 1
            print(f'# Communication round {self.current_communication_round} #')

            self.global_train_step()

            if self.scheduler and not self.scheduler.after_val:
                self.scheduler.step()
                self.global_history[('LR', self.current_communication_round)] = self.scheduler.lr()
                if self.config.training.globally_derived_lr:
                    self.model_trainer.config.training.optimizer.lr = self.scheduler.lr() * self.config.training.globally_derived_lr.factor


            if global_epoch % self.config.training.glob.validation_frequency == 0:
                if self.config.data.use_val:
                    self.evaluate('VAL') # evaluate global model on validation data
                self.evaluate('TST') # evaluate global model on test data

                if self.current_communication_round in self.config.training.glob.save_rounds or self.current_communication_round == communication_rounds:
                    self.save(self.checkpoint_path+f'_CR{self.current_communication_round}')
                
                #EarlyStopping
                if self.early_stopper:
                    self.early_stopper(self.global_history[(self.early_stopper.metric, self.current_communication_round)])

                    if self.early_stopper.improved:
                        self.save(self.checkpoint_path, best=True)
                    if self.early_stopper.stop:
                        print(f'Early stopping the global federated training since we had no improvement of {self.early_stopper.metric} for {self.early_stopper.patience} rounds. Training was stopped after {self.current_communication_round} CommunicationRounds.')
                        break
                else:
                    self.save(self.checkpoint_path, best=True)

                if self.scheduler and self.scheduler.after_val:
                    if hasattr(self.scheduler, 'metric'):
                        self.scheduler.step(self.global_history.get(self.scheduler.metric))
                    else:
                        self.scheduler.step()
                    if self.config.training.globally_derived_lr:
                        self.model_trainer.config.training.optimizer.lr = self.scheduler.lr() * self.config.training.globally_derived_lr.factor
                    self.global_history[('LR', self.current_communication_round)] = self.scheduler.lr()

            self.log_status()

            if self.debug and self.current_communication_round >= 2:
                break
        
        print(f'### {self.__class__.__name__} Training (END) ###')
    
        
    def evaluate(self, split='VAL', *args, **kwargs):
        client_metrics = defaultdict(list)
        client_samples = []
        client_ids = [] #also track this so we can be sure to have the right order if required
        for client_id, client in self.clients.items():
            local_metrics = client.evaluate(self.current_communication_round, split, self.global_model)
            for metric, value in local_metrics.wandb_dict(step_key='CommunicationRound').items():
                client_metrics[metric].append(value)
            client_samples.append(client.get_sample_number(split))
            client_ids.append(client_id)
        for metric, values in client_metrics.items():
            self.global_history[(metric, self.current_communication_round)] = np.mean(values)
            if metric in ['CommunicationRound', 'LocalEpoch', 'OverallTrainedEpochs']:
                continue
            self.global_history[(metric+'_WEIGHTED', self.current_communication_round)] = np.average(values, weights=client_samples)
        return client_metrics, client_samples, client_ids

    @timer
    def global_train_step(self, *args, **kwargs):
        local_weights = []
        for client_id, client in self.clients.items():
            client.sent_models += 1
            client.received_models += 1
            w_local = client.train(self.global_model, self.current_communication_round)
            if self.config.training.weigh_sample_quantity:
                local_weights.append((client.get_sample_number('TRN'), copy.deepcopy(w_local)))
            else:
                local_weights.append((1, copy.deepcopy(w_local)))

        # update global weights
        self.global_model = copy.deepcopy(self.aggregate(local_weights))

        self.total_num_exchanged_models += 2*self.num_clients
        self.global_history[('TotalExchangedModels', self.current_communication_round)] = self.total_num_exchanged_models


    def aggregate(self, local_weights, *args, **kwargs):
        training_num = 0
        for idx in range(len(local_weights)):
            (sample_num, averaged_params) = local_weights[idx]
            training_num += sample_num

        (sample_num, averaged_params) = local_weights[0]
        for k in averaged_params.keys():
            for i in range(0, len(local_weights)):
                local_sample_number, local_model_params = local_weights[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
    

    def log_status(self, *args, **kwargs):
        metric_dict = self.global_history.wandb_dict(step=self.current_communication_round, step_key='CommunicationRound')
        metric_strings = [f'{metric}: {value}' for metric, value in metric_dict.items() if metric not in ['CommunicationRound']]

        ms = ' | '.join(metric_strings)
        print(f'# Model performance in communication round {self.current_communication_round}: {ms} #')

    def summary(self):
        print('### Global History Summary ###')
        #self.global_history.summary(step_key='CommunicationRound', mode='latest')
        key_value = 'VAL_' + self.loss.name  if self.config.data.use_val else 'TST_' + self.loss.name
        key_value += '_L'
        self.global_history.summary(key_value=key_value, max_key=self.loss.subject_to=='max', step_key='CommunicationRound', mode='best')

    def save_histories(self):
        self.global_history.save()
        self.global_history.to_csv()
        for client in self.clients.values():
            client.local_history.save()
            client.local_history.to_csv()

    def save(self, path=None, best=False):
        self.save_histories()
        assert self.checkpoint_path or path, 'If the Environment object has no default checkpoint_path you have to provide a path!'
        path = self.checkpoint_path if path is None else path
        if best:
            path += '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        torch.save(self.global_model, statepath)

    def load(self, path=None, best=False):
        assert self.checkpoint_path or path, 'If the Environment object has no default checkpoint_path you have to provide a path!'
        path = self.checkpoint_path if path is None else path
        if best:
            path += '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        self.global_model = torch.load(statepath)
        print(f'Loaded global model state from {path}')        

    def load_best(self):
        self.load(best=True)

    def load_histories(self):
        print('### Loading Histories ###')
        self.global_history.load()
        for client in self.clients.values():
            client.local_history.load()

    def get_history(self):
        self.global_history.load()
        return self.global_history
    
    def get_client_histories(self):
        for client in self.clients.values():
            client.local_history.load()
        return {client_id: client.local_history for client_id, client in self.clients.items()}