import logging
from numbers import Number
from typing import Iterable

from prettytable import PrettyTable
import dill
import pandas as pd
import numpy as np

class History(object):
    """
    A class for managing and tracking experimental history.

    Args:
        keys (list): List of keys to track (e.g., ['Metric', 'Epoch', 'Batch']).
        values (list): List of values to track (e.g., ['Value']).
        primary_key (str): The primary key to use (e.g., 'Metric').
        savefile (str, optional): The file path for saving the history object. Defaults to None.

    Methods:
        (See code for detailed method descriptions)
    """
    def __init__(self, keys = ['Metric', 'Epoch', 'Batch'], values = ['Value'], primary_key = 'Metric', savefile = None):
        """
        Initializes a History object.

        Parameters:
            keys (list, optional): List of keys to track in the history. Default is ['Metric', 'Epoch', 'Batch'].
            values (list, optional): List of value fields to track. Default is ['Value'].
            primary_key (str, optional): The primary key to be used. Default is 'Metric'.
            savefile (str, optional): The path to save the history data. Default is None.
        """
        super().__init__()
        assert len(keys) > 0 and len(values) > 0, 'A history object needs to track at least one key and one value field! '
        assert primary_key in keys, f'{primary_key} has to be a part of keys!'
        self._keys = keys
        self._values = values
        self.primary_key = primary_key
        self.pk_pos = keys.index(primary_key)
        self.df = pd.DataFrame(columns=self._keys+self._values)
        self.savefile = savefile
        self.allowed_modes = ['best', 'latest', 'last'] # For summaries

    def __contains__(self, element):
        """
        Checks if the specified element is present in the history.

        Parameters:
            element (str or list): The element(s) to check for.

        Returns:
            bool: True if the element is present, False otherwise.
        """
        if isinstance(element, str):
            element = [element]
        assert len(element) == len(self._keys), f'Element has to be an Iterable of len {len(self._keys)} {self._keys} but got {element}! Please provide an Iterable with {len(self._keys)} elements!'
        return eval(self.eval_str(element, self._keys)+'.any()')

    def __setitem__(self, index, value):
        """
        Sets the value for the specified index.

        Parameters:
            index (str or list): The index to set.
            value (Number or Iterable): The value to set.

        Raises:
            ValueError: If the value is not a valid type.
        """
        if isinstance(index, str):
            index = [index]
        assert len(index) == len(self._keys), f'index has to be an Iterable of len {len(self._keys)} {self._keys} but got {index}! Please provide an Iterable with {len(self._keys)} elements!'
        if isinstance(value, Number):
            assert len(self._values) == 1, f'This history object tracks more than one value field: {self._values}! Please provide an Iterable with {len(self._values)} elements!'
            element = [*index, value]
        elif isinstance(value, Iterable):
            assert len(value) == len(self._values), f'This history object tracks {len(self._values)} value fields: {self._values}! Please provide an Iterable with {len(self._values)} elements!'
            element = [*index, *value]
        else:
            raise ValueError(f'This history object only handles single values or iterables of values that must match with the initialized values: {self._values}!')

        if index not in self:
            self.df.loc[len(self.df)] = element
        else:
            self.df.loc[self.index(index)] = element

    def __call__(self, index, value):
        """
        Calls the __setitem__ method for setting values.

        Parameters:
            index (str or list): The index to set.
            value (Number or Iterable): The value to set.

        Raises:
            ValueError: If the value is not a valid type.
        """
        self.__setitem__(index, value)

    def __getitem__(self, index):
        
        if isinstance(index, str):
            index = [index]
        assert len(index) == len(self._keys), f'index has to be an Iterable of len {len(self._keys)} {self._keys} but got {index}! Please provide an Iterable with {len(self._keys)} elements!'
        item = list(self.df[eval(self.eval_str(index, self._keys))][self._values].itertuples(index=False, name=None))[-1]
        if len(self._values) == 1:
            item = item[0]
        return item

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return self.df.__repr__()

    def eval_str(self, index, keys):
        """
        Generates an evaluation string for index and keys.

        Parameters:
            index (str or list): The index to evaluate.
            keys (str or list): The keys for evaluation.

        Returns:
            str: The generated evaluation string.
        """
        if isinstance(index, str):
            index = [index]
        if isinstance(keys, str):
            keys = [keys]
        assert len(index) == len(keys), f'Length and order of index and keys must match! {len(index)}!={len(keys)}'

        eval_list = []
        for i, key in zip(index, keys):
            eq = f'"{i}"' if isinstance(i, str) else i
            eval_list.append(f'(self.df.{key} == {eq})')
        eval_str = '(' + '&'.join(eval_list) + ')'
        return eval_str

    def get(self, index, keys=None, values=None, only_last=True):
        """
        Gets values based on index and keys.

        Parameters:
            index (str or list): The index to get.
            keys (str or list, optional): The keys to use. Default is primary key.
            values (str or list, optional): The values to get. Default is tracked values.
            only_last (bool, optional): If True, returns only the last item. Default is True.

        Returns:
            list or value: The requested values.
        """
        keys = self.primary_key if keys is None else keys
        values = self._values if values is None else values
        if isinstance(index, str):
            index = [index]
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(values, str):
            values = [values]
        s = self.eval_str(index, keys)
        x = self.df[eval(s)]
        item = list(x[values].itertuples(index=False, name=None))
        if only_last:
            item = item[-1]
        if len(values) == 1:
            if only_last:
                item = item[0]
            else:
                item = [i[0] for i in item]
        return item

    def remove(self, element):
        """
        Removes an element from the history.

        Parameters:
            element (str or list): The element(s) to remove.

        Returns:
            list or None: The removed element or None if not found.
        """
        if element in self:
            idx = self.index(element)
            row = list(self.df.iloc[idx])
            self.df.drop(index=idx, inplace=True)
            self.df.reset_index(inplace=True, drop=True)
            return row
        else:
            logging.warning(f'History does not contain {element}!')
            return None

    def unique(self, key=None):
        """
        Gets unique values for a specified key.

        Parameters:
            key (str or list, optional): The key(s) to get unique values. Default is primary key.

        Returns:
            list: The unique values.
        """
        key = self.primary_key if key is None else key
        if isinstance(key, str):
            uniques = list(self.df[key].unique())            
        else:
            if len(key) == 1:
                uniques = list(self.df[key[0]].unique())
            else:
                uniques = [list(self.df[k].unique()) for k in key]
        return uniques

    def index(self, element):
        """
        Gets the index of an element in the history.

        Parameters:
            element (str or list): The element(s) to get the index for.

        Returns:
            int or None: The index or None if not found.
        """
        if element in self:
            return self.df[eval(self.eval_str(element, self._keys))].index[0]
        else:
            return None

    def key_names(self):
        """
        Gets the names of the keys.

        Returns:
            list: The names of the keys.
        """
        return self._keys

    def value_names(self):
        """
        Gets the names of the values.

        Returns:
            list: The names of the values.
        """
        return self._values
    
    def keys(self):
        """
        Gets the keys.

        Returns:
            list: The keys.
        """
        if len(self._keys) > 1:
            return [v for v in self.df[self._keys].itertuples(index=False, name=None)]
        else:
            return [v[0] for v in self.df[self._keys].itertuples(index=False, name=None)]
        
    def values(self):
        """
        Gets the values.

        Returns:
            list: The values.
        """
        if len(self._values) > 1:
            return [v for v in list(self.df[self._values].itertuples(index=False, name=None))]
        else:
            return [v[0] for v in list(self.df[self._values].itertuples(index=False, name=None))]
            
    def items(self):
        """
        Gets the items.

        Returns:
            zip: The zipped items.
        """
        return zip(self.keys(), self.values())

    def metrics(self):
        """
        Gets the tracked metrics.

        Returns:
            list or None: The tracked metrics or None if Metric key is not tracked.
        """
        if 'Metric' in self._keys:
            return sorted(self.df.Metric.unique())
        else:
            logging.warning('This History object does not track a Metric key!')
            return None

    def wandb_dict(self, key=None, step=None, step_key='Epoch', key_suffix=None):
        """
        Generates a dictionary for logging with W&B.

        Parameters:
            key (str or list, optional): The key(s) to use. Default is primary key.
            step (int, optional): The step to use. Default is maximum step.
            step_key (str, optional): The step key to use. Default is 'Epoch'.
            key_suffix (str, optional): The key suffix to use. Default is None.

        Returns:
            dict or None: The generated dictionary or None if step_key is not tracked.
        """
        if step_key in self._keys:
            key = self.primary_key if key is None else key
            step = max(self.unique(step_key)) if step is None else step
            d = {step_key: step}
            for k in self.unique(key):
                try:
                    new_key = k if key_suffix is None else k + '_' + key_suffix
                    v = self.get((k, step), (key, step_key))
                    d[new_key] = v
                except:
                    logging.warning(f'Did not find {step_key} {step} for {key} {k}! Skipping!')
            return d
        else:
            logging.warning(f'Cannot provide wandb data for a specific step since this History object does not track {step_key} as a step key (e.g. Epoch, Batch, Step, LocalEpoch, CommunicationRound)!') 
            return None

    def summary(self, key_value=None, max_key=True, step_key='Epoch', mode='best', verbose=True):
        """
        Generates a summary of history data.

        Parameters:
            key_value (str, optional): The key value to use for mode 'best'. Default is None.
            max_key (bool, optional): If True, uses argmax for 'best' mode, else argmin. Default is True.
            step_key (str, optional): The step key to use. Default is 'Epoch'.
            mode (str, optional): The summary mode ('best', 'latest', or 'last'). Default is 'best'.
            verbose (bool, optional): If True, prints summary. Default is True.

        Returns:
            dict or None: The summary dictionary or None if step_key is not tracked.
        """
        assert mode in self.allowed_modes, f'mode must be one of {self.allowed_modes}, got {mode} instead!'
        if step_key in self._keys:
            if mode == 'best':
                key_value = self.unique(self.primary_key)[0] if key_value is None else key_value
                best_step = self.argmax(key_value, index_key=step_key) if max_key else self.argmin(key_value, index_key=step_key)
                summary_dict = self.wandb_dict(step=best_step, step_key=step_key)
            elif mode in ['latest', 'last']:
                summary_dict = self.wandb_dict(step_key=step_key)
            else:
                summary_dict = None
        else:
            if verbose:
                logging.warning(f'Cannot provide summary since this History object does not track {step_key} as a step key (e.g. Epoch, Batch, Step, LocalEpoch, CommunicationRound)!') 
            summary_dict = None
        if summary_dict is not None:
            if mode == 'best':
                summary_string = f'History summary for the {mode} ({key_value}) {step_key}:\n'
            else:
                summary_string = f'History summary for the {mode} {step_key}:\n'
            t = PrettyTable(['Metric', 'Value'])
            t.add_row([step_key, summary_dict[step_key]])
            for k, v in summary_dict.items():
                if k == step_key:
                    continue
                t.add_row([k, round(v, 6)])
            t.align['Metric'] = 'l'
            t.align['Value'] = 'r'
            summary_string += t.get_string()
            if verbose:
                logging.info(summary_string)
        return summary_dict

    def save(self, path=None):
        """
        Saves the History object to a file.

        Parameters:
            path (str, optional): The file path. Default is None.

        Returns:
            None
        """
        assert self.savefile or path, 'If the History object has no default savefile you have to provide a path!'
        path = self.savefile if path is None else path
        path = path if path.endswith('.history') else path + '.history'
        with open(path, 'wb') as file:
            dill.dump(self.__dict__, file)
    
    def load(self, path=None, force = False):
        """
        Loads a History object from a file.

        Parameters:
            path (str, optional): The file path. Default is None.
            force (bool, optional): If True, forces the loading even without file extension. Default is False.

        Returns:
            None
        """
        assert self.savefile or path, 'If the History object has no default savefile you have to provide a path!'
        if not force:
            path = self.savefile if path is None else path
            path = path if path.endswith('.history') else path + '.history'
        with open(path, 'rb') as file:
            self.__dict__ = dill.load(file)

    def to_csv(self, path=None):
        """
        Saves the History object to a CSV file.

        Parameters:
            path (str, optional): The file path. Default is None.

        Returns:
            None
        """
        assert self.savefile or path, 'If the History object has no default savefile you have to provide a path!'
        path = self.savefile if path is None else path
        path = path if path.endswith('.csv') else path + '.csv'
        self.df.to_csv(path, index=False)

    def from_csv(self, path=None):
        """
        Loads a History object from a CSV file.

        Parameters:
            path (str, optional): The file path. Default is None.

        Returns:
            None
        """
        assert self.savefile or path, 'If the History object has no default savefile you have to provide a path!'
        path = self.savefile if path is None else path
        path = path if path.endswith('.csv') else path + '.csv'
        self.df = pd.read_csv(path)
        logging.warning('This implementation does not guarantee that the History object state is consistent with the DataFrame loaded from the csv file! If you want to save and load the whole History object state use save() and load() instead!')


    def update(self, other, key_suffix=None):
        """
        Updates the History object with values from another object.

        Parameters:
            other (dict): The other object with values.
            key_suffix (str, optional): The key suffix to add. Default is None.

        Returns:
            None
        """
        for key, value in other.items():
            new_key = list(key)
            if key_suffix is not None:
                new_key[self.pk_pos] = new_key[self.pk_pos] + '_' + key_suffix
            self[new_key] = value

    def add_col(self, name, default_value, type='key'):
        """
        Adds a column to the History object.

        Parameters:
            name (str): The column name.
            default_value: The default value.
            type (str, optional): The type ('key' or 'value'). Default is 'key'.

        Returns:
            None
        """
        if type == 'key':
            self._keys.append(name)
        else:
            self._values.append(name)
        self.df[name] = default_value
        self.df = self.df.reindex(self._keys+self._values, axis=1)

    def remove_cols(self, names):
        """
        Removes columns from the History object.

        Parameters:
            names (list): The list of column names.

        Returns:
            None
        """
        for name in names:
            if name in self._keys:
                self._keys.remove(name)
            elif name in self._values:
                self._values.remove(name)
            else:
                logging.warning(f'Column {name} not found!')
            self.df.drop(name, axis=1, inplace=True)

    def aggregate(self, aggregation_fn, index=None, values=None):
        """
        Aggregates data in the History object.

        Parameters:
            aggregation_fn (function): The aggregation function.
            index (str or list, optional): The index to use. Default is tracked primary key.
            values (str or list, optional): The values to use. Default is first tracked value.

        Returns:
            dict: The aggregated data.
        """
        tracked = self.unique(self.primary_key)
        if index is None:
            index = tracked
        if values is None:
            values = self._values[:1]
        if isinstance(values, str):
            assert values in self._keys + self._values, f'{values} is not tracked by this History object! Try one of {self._keys+self._values}'
        else:
            assert isinstance(values, Iterable)
            for v in values:
               assert v in self._keys + self._values, f'{v} is not tracked by this History object! Try one of {self._keys+self._values}'

        if isinstance(index, str):
            assert index in self.unique(self.primary_key), f'{index} is not tracked by this History object! Try one of {tracked}'
            result = aggregation_fn(self.get(index, self.primary_key, values, only_last=False))
        else:
            assert(isinstance(index, Iterable))
            for i in index:
                assert i in self.unique(self.primary_key), f'{i} is not tracked by this History object! Try one of {tracked}'
            result = {}
            for i in index:
                result[i] = aggregation_fn(self.get(i, self.primary_key, values, only_last=False))
        return result

    def arg_aggregation(self, aggregation_fn, index=None, values=None, index_key='Epoch'):
        """
        Aggregates data in the History object and returns the index key.

        Parameters:
            aggregation_fn (function): The aggregation function.
            index (str or list, optional): The index to use. Default is tracked primary key.
            values (str or list, optional): The values to use. Default is first tracked value.
            index_key (str or list, optional): The index key(s) to use. Default is 'Epoch'.

        Returns:
            dict or value: The aggregated data for the index key(s).
        """
        if isinstance(index_key, str):
            index_key = [index_key]
        for ik in index_key:
            assert ik in self._keys, f'The index_key {ik} has to be part of self._keys ({self._keys})!'
        if values is None:
            values = self._values[:1]
        _aggregate = self.aggregate(aggregation_fn, index, values + index_key)        
        if isinstance(_aggregate, dict):
            results = {}
            for key, value in _aggregate.items():
                result_value = value[len(values):]
                if isinstance(result_value, Iterable) and len(result_value) == 1:
                    result_value = result_value[0]
                results[key] = result_value
        else:
            results = _aggregate[len(values):]
            if isinstance(results, Iterable) and len(results) == 1:
                results = results[0]
        return results

    def max(self, index=None, values=None):
        return self.aggregate(max, index, values)

    def min(self, index=None, values=None):
        return self.aggregate(min, index, values)

    def mean(self, index=None, values=None):
        return self.aggregate(np.mean, index, values)

    def std(self, index=None, values=None):
        return self.aggregate(np.std, index, values)

    def argmax(self, index=None, values=None, index_key='Epoch'):
        return self.arg_aggregation(max, index, values, index_key)

    def argmin(self, index=None, values=None, index_key='Epoch'):
        return self.arg_aggregation(min, index, values, index_key)

    def concat(self, other):
        self.df = self + other
 
    def __add__(self, other):
        self.df = pd.concat([self.df, other.df])
        return self

def merge(histories):
    """
    Merge multiple History objects.

    Args:
        histories (list): List of History objects to merge.

    Returns:
        History: The merged History object.

    """
    agg_hist = histories[0]
    for h in histories[1:]:
        agg_hist += h
    agg_hist.df = agg_hist.df.reset_index().drop('index', axis=1)
    return agg_hist