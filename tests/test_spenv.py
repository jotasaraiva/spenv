from spenv import mvspec, specenv
import pandas as pd
import numpy as np

def test_mvspec():
    """Test multivariate spectral estimation."""
    data = pd.read_csv("tests/nyse.csv")
    assert mvspec(data.value)['freq'].size == 1000, "Frequency estimation is incorrect"

def test_specenv():
    """Test spectral envelope."""
    data = pd.read_csv("tests/nyse.csv")["value"].values.reshape([-1, 1])
    xdata = np.concatenate([data, np.abs(data), data**2], axis=1)
    res = specenv(xdata).shape[0]
    assert res == 1000, "Spectral envelope frequencies are different sizes"


