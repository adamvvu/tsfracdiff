[![Build](https://img.shields.io/github/workflow/status/adamvvu/tsfracdiff/Unit%20Tests?style=for-the-badge)](https://github.com/adamvvu/tsfracdiff/actions/workflows/tsfracdiff_tests.yml)
[![PyPi](https://img.shields.io/pypi/v/tsfracdiff?style=for-the-badge)](https://pypi.org/project/tsfracdiff/)
[![Downloads](https://img.shields.io/pypi/dm/tsfracdiff?style=for-the-badge)](https://pypi.org/project/tsfracdiff/)
[![License](https://img.shields.io/pypi/l/tsfracdiff?style=for-the-badge)](https://github.com/adamvvu/tsfracdiff/blob/master/LICENSE)

Efficient and easy to use fractional differentiation transformations for
stationarizing time series data in Python.

------------------------------------------------------------------------

## **tsfracdiff**

Data with high persistence, serial correlation, and non-stationarity
pose significant challenges when used directly as predictive signals in
many machine learning and statistical models. A common approach is to
take the first difference as a stationarity transformation, but this
wipes out much of the information available in the data. For datasets
where there is a low signal-to-noise ratio such as financial market
data, this effect can be particularly severe. Hosking (1981) introduces
fractional (non-integer) differentiation for its flexibility in modeling
short-term and long-term time series dynamics, and López de Prado (2018)
proposes the use of fractional differentiation as a feature
transformation for financial machine learning applications. This library
is an extension of their ideas, with some modifications for efficiency
and robustness.

[Documentation](https://adamvvu.github.io/tsfracdiff/docs/)

## Getting Started

### Installation

`pip install tsfracdiff`

#### Dependencies:

    # Required
    python3 # Python 3.6+
    numpy
    pandas
    arch    # If on Python 3.6, use arch <= v4.17

    # Suggested
    joblib

### Usage

``` python
# A pandas.DataFrame/np.array with potentially non-stationary time series
df 

# Automatic stationary transformation with minimal information loss
from tsfracdiff import FractionalDifferentiator
fracDiff = FractionalDifferentiator()
df = fracDiff.FitTransform(df)
```

For a more in-depth example, see this
[notebook](https://adamvvu.github.io/tsfracdiff/examples/Example.html).

## References

Hosking, J. R. M. (1981). Fractional Differencing. Biometrika, 68(1),
165--176. <https://doi.org/10.2307/2335817>

López de Prado, Marcos (2018). Advances in Financial Machine Learning.
John Wiley & Sons, Inc.
