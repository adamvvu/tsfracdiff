![Unit
Tests](https://github.com/AdamWLabs/tsfracdiff/actions/workflows/tsfracdiff_tests.yml/badge.svg?branch=master)

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

### Documentation/Examples

For a more in-depth example, see the notebook in `/examples`. See
`/docs` for documentation.

## References

Hosking, J. R. M. (1981). Fractional Differencing. Biometrika, 68(1),
165--176. <https://doi.org/10.2307/2335817>

López de Prado, Marcos (2018). Advances in Financial Machine Learning.
John Wiley & Sons, Inc.
