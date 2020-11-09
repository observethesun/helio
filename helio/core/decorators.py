"""Contains decorators."""
import os
import inspect
import concurrent.futures as cf
from textwrap import dedent
import functools
from functools import wraps
import numpy as np
from numpydoc.docscrape import NumpyDocString

from .sampler import R

def partialmethod(func, *frozen_args, **frozen_kwargs):
    """Wrap a method with partial application of given positional
    and keyword arguments.

    Parameters
    ----------
    func : callable
        A method to wrap.
    frozen_args : misc
        Fixed positional arguments.
    frozen_kwargs : misc
        Fixed keyword arguments.

    Returns
    -------
    method : callable
        Wrapped method.
    """
    @functools.wraps(func)
    def method(self, *args, **kwargs):
        """Wrapped method."""
        return func(self, *frozen_args, *args, **frozen_kwargs, **kwargs)
    return method

TEMPLATE_DOCSTRING = """
    Apply {description} to batch data.

    Parameters
    ----------
    src : str
        Attribute to get data from. Data from this attribute will be passed to
        the first argument of ``{full_name}``.
    dst : str, optional
        Attribute to put data in. Will be same as input if not specified.
    {kwargs}

    Returns
    -------
    batch : Batch
        {returns}
"""
TEMPLATE_DOCSTRING = dedent(TEMPLATE_DOCSTRING).strip()


def execute(how, split_attrs=True):
    """Execute scheme decorator."""
    def decorator(method):
        """Returned decorator."""
        @wraps(method)
        def wrapper(self, *args, src=None, dst=None, **kwargs):
            """Method wrapper."""
            src = np.atleast_1d(src)
            dst = src if dst is None else np.atleast_1d(dst)
            if len(dst) != len(src) and split_attrs:
                if len(src) == 1:
                    src = np.repeat(src, len(dst))
                elif len(dst) == 1:
                    dst = np.repeat(dst, len(src))
                else:
                    raise ValueError('Mismatch of src and dst lenghts.')

            for d in dst:
                if d not in self.data:
                    self.data[d] = np.array([None] * len(self))
                if d not in self.meta:
                    self.meta[d] = np.array([None] * len(self))

            results = []
            n_workers = kwargs.get('n_workers', os.cpu_count() * 4)
            kwargs.pop('n_workers', None)

            random_kwargs = {k: v(size=len(self)) for k, v in kwargs.items() if isinstance(v, R)}
            static_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, R)}

            if not split_attrs:
                src = [src]
                dst = [dst]

            if how == 'threads':
                with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    for s, d in zip(src, dst):
                        for i in range(len(self)):
                            ikwargs = dict(static_kwargs, **{k: v[i] for k, v in random_kwargs.items()})
                            res = executor.submit(method, self, i, *args, src=s, dst=d, **ikwargs)
                            futures.append(res)
                    cf.wait(futures, timeout=None, return_when=cf.ALL_COMPLETED)

                results = [f.result() for f in futures]
                if any(isinstance(res, Exception) for res in results):
                    errors = [error for error in results if isinstance(error, Exception)]
                    print(errors)

            elif how == 'loop':
                for s, d in zip(src, dst):
                    for i in range(len(self)):
                        ikwargs = dict(static_kwargs, **{k: v[i] for k, v in random_kwargs.items()})
                        results.append(method(self, i, *args, src=s, dst=d, **ikwargs))
            else:
                raise ValueError('Unknown execution scheme {}.'.format(how))

            return self
        return wrapper
    return decorator

def extract_actions(module, first_arg):
    """Extract callable attributes that have ``first_arg`` as a first parameter
    from a module."""
    actions_dict = {}
    arg = None
    for (k, v) in module.__dict__.items():
        if callable(v):
            try:
                arg = inspect.getfullargspec(v).args[0]
            except (TypeError, IndexError):
                continue
            if arg == first_arg:
                method = {k: (v, '.'.join([module.__name__, k]), k)}
                actions_dict.update(method)
    return actions_dict

def add_actions(actions_dict, template_docstring):
    """Add new actions in a class.

    Parameters
    ----------
    actions_dict : dict
        A dictionary, containing names of new methods as keys and a tuple of callable,
        full name and description for each method as values.

    template_docstring : str
        A string that will be formatted for each new method from ``actions_dict``.

    Returns
    -------
    decorator : callable
        Class decorator.
    """
    def decorator(cls):
        """Returned decorator."""
        for method_name, (func, full_name, description) in actions_dict.items():
            nds = NumpyDocString(func.__doc__)
            params = nds["Parameters"]
            params = '\n'.join(params[0][-1][1:])
            returns = nds["Returns"][0][-1][0]
            docstring = template_docstring.format(full_name=full_name,
                                                  description=description,
                                                  kwargs=params,
                                                  returns=returns)
            method = partialmethod(cls.apply, func)
            method.__doc__ = docstring
            setattr(cls, method_name, method)
        return cls
    return decorator
