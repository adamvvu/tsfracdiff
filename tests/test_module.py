from tsfracdiff import *
import numpy as np
import pandas as pd
np.random.seed(42)
import pytest

def _GenerateData():

    T = 1000
    K = 5
    
    df = [ np.array([1 for k in range(K)]) ]
    mu = np.random.normal(0, 0.25, size=(K))
    for t in range(T):
        d_t = mu + df[-1] + np.random.normal(0, 1, size=(K))
        df.append( d_t )
    df = pd.DataFrame(np.vstack(df))
    return df

def _TestStationary( df_frac, fracDiff ):
    if isinstance(df_frac, pd.DataFrame):
        for k in range(df_frac.shape[1]):
            assert fracDiff.UnitRootTest.IsStationary( df_frac.iloc[:,k].dropna() )
    elif isinstance(df_frac, np.ndarray):
        for k in range(df_frac.shape[1]):
            assert fracDiff.UnitRootTest.IsStationary( df_frac[:,k].dropna() )
    else:
        raise Exception('Invalid datatype returned.')

    return

def _TestFracDiff( df, unitRootTest, parallel=True ):
    fracDiff = FractionalDifferentiator(unitRootTest=unitRootTest)
    df_frac = fracDiff.FitTransform( df, parallel=parallel )
    _TestStationary( df_frac, fracDiff )
    return df_frac

def _TestAutoFracDiff( df, unitRootTest ):
    """
    Test automatic fit-transform
    """
    df_frac_par = _TestFracDiff( df, unitRootTest=unitRootTest, parallel=True )
    df_frac_seq = _TestFracDiff( df, unitRootTest=unitRootTest, parallel=False )
    assert np.allclose(df_frac_par.values, df_frac_seq.values, equal_nan=True)
    print('AutoFracDiff: OK')
    return

def _TestInvTransform( df, unitRootTest ):
    """
    Test inverse-transform
    """
    fracDiff = FractionalDifferentiator(unitRootTest=unitRootTest)
    df_frac = fracDiff.FitTransform( df )
    df_inv = fracDiff.InverseTransform( df_frac )
    assert np.allclose(df.values, df_inv.values, equal_nan=True)
    print('InvTransform: OK')
    return

def test_RunAllTests():

    df = _GenerateData()

    unitRootTests = ['PP', 'ADF']
    for unitRootTest in unitRootTests:
        print(f'Testing {unitRootTest}')
        _TestAutoFracDiff( df, unitRootTest=unitRootTest )
        _TestInvTransform( df, unitRootTest=unitRootTest )

    return