import argparse
import numpy as np

from pathvalidate import ValidationError, validate_filename
import os.path

def checkMode(mode: str):
    """
    Check if mode is "0" or "1"
    Keyword arguments:
        mode -- Must be "0" or "1" otherwise an exception will be thrown
    Return:
        mode 
    """
    if mode != "0" and mode != "1":
        raise argparse.ArgumentTypeError("Invalid mode option. Use \"0\" = game window or \"1\" = game window with plots")

    return mode


def is_valid_name(path: str):

    try:
        validate_filename(path)

    except ValidationError:
        raise argparse.ArgumentTypeError("Invalid gif path/name: This file path/name is not valid")
    
    return path 


def check_filePath(path: str):

    path_tmp = path + ".index"
    if not os.path.isfile(path_tmp):
        raise argparse.ArgumentTypeError("The path to the model weight's does not exists.")
    
    return path 


def check_step_num(num: str):

    try:
        num = int(num)
    except:
        raise argparse.ArgumentTypeError("The step number must be an integer.")
    

    if num <= 0:
        raise argparse.ArgumentTypeError("The step number must be positive and greater than zero")
    
    return num



class dummy_context_mgr():
    """
    A null object required for a conditional with statement
    """
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence