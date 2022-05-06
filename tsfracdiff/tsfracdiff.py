from .unit_root_tests import *

import pandas as pd
import numpy as np

class FractionalDifferentiator:
    
    def __init__(self, maxOrderBound=1, significance=0.01, precision=0.01, 
                       unitRootTest='PP', unitRootTestConfig={}):
        """
        Provides estimation of the minimum fractional order of differentiation required for stationarity
        and data transformations.
        
        The available stationarity/unit root tests are:
        -----------------------------------------------
            - 'PP'  : Phillips and Perron (1988) [default]
            - 'ADF' : Augmented Dickey-Fuller (Said & Dickey, 1984)

        Parameters:
        -----------
            maxOrderBound       (float) Maximum real-valued order to search in (0, maxOrderBound)
            significance        (float) Statistical significance level
            precision           (float) Precision of estimated order
            unitRootTest        (str)   Unit-root/stationarity tests: ['PP','ADF']
            unitRootTestConfig  (dict)  Optional keyword arguments to pass to unit root tests

        Attributes:
        -----------
            orders              (list)  Estimated minimum orders of differentiation

        Example:
        --------
        ```
	    # A pandas.DataFrame/np.array with potentially non-stationary time series
        df 
	
	    # Automatic stationary transformation with minimal information loss
        from tsfracdiff import FractionalDifferentiator
        fracDiff = FractionalDifferentiator()
        df = fracDiff.fit_transform(df)
        ```
        """
        self.maxOrderBound = maxOrderBound
        self.significance = significance
        self.precision = precision
        
        # Critical value checks
        checkCV = False
        cv_sig = None
        if (self.significance in [0.01, 0.05, 0.1]):
            checkCV = True
            cv_sig = str(int(self.significance * 100)) + '%'
        
        # Unit-root/Stationarity tests
        if unitRootTest == 'PP':
            self.UnitRootTest = PhillipsPerron(significance=significance, checkCV=checkCV, cv_sig=cv_sig)
        elif unitRootTest == 'ADF':
            self.UnitRootTest = ADFuller(significance=significance, checkCV=checkCV, cv_sig=cv_sig)
        else:
            raise Exception('Please specify a valid unit root test.')
        self.UnitRootTest.config.update( unitRootTestConfig )

        # States
        self.isFitted = False
        self.orders = []
        self.lagData = None
        
    def Fit(self, df, parallel=True):
        """
        Estimates the minimum orders of differencing required for stationarity.
        
        Parameters:
        -----------
            df       (pandas.DataFrame/np.array) Raw data
            parallel (bool) Use multithreading if true (default). Requires `joblib`.
        """
        df = pd.DataFrame(df).sort_index()
        
        # Estimate minimum order of differencing
        if parallel:
            try:
                import multiprocessing
                from joblib import Parallel, delayed
                from functools import partial
            except ImportError:
                raise Exception('The module `joblib` is required for parallelization.')

            def ApplyParallel(df, func, **kwargs):
                n_jobs = min(df.shape[1], multiprocessing.cpu_count())
                res = Parallel(n_jobs=n_jobs, prefer='threads')( delayed(partial(func, **kwargs))(x) for x in np.array_split(df, df.shape[1], axis=1) )
                return res
            orders = ApplyParallel(df, self._MinimumOrderSearch, upperOrder=self.maxOrderBound, first_run=True)
        else:
            orders = []
            for j in range(df.shape[1]):
                orders.append( self._MinimumOrderSearch(df.iloc[:,j], upperOrder=self.maxOrderBound, first_run=True) )
        self.orders = orders
        
        # Store lagged data for inverse-transformations
        numLags = [ (len(self._GetMemoryWeights(order)) - 1) for order in self.orders ]
        self.lagData = [ df.iloc[:,j].head(lag) for j,lag in enumerate(numLags) ]
        
        self.isFitted = True

        return
        
    def FitTransform(self, df, parallel=True):
        """
        Estimates the minimum orders and returns a fractionally differentiated dataframe.
        
        Parameters
        ----------
            df       (pandas.DataFrame/np.array) Raw data
            parallel (bool) Use multithreading if true (default). Requires `joblib`.
        """
        if not self.isFitted: 
            self.Fit(df, parallel=parallel)
        fracDiffed = self.Transform(df)
        return fracDiffed
    
    def Transform(self, df):
        """
        Applies a fractional differentiation transformation based on estimated orders.

        Parameters
        ----------
            df  (pandas.DataFrame/np.array) Raw data
        """
        if not self.isFitted: 
            raise Exception('Fit the model first.')
            
        df = pd.DataFrame(df).sort_index()
        fracDiffed = []
        for j in range(df.shape[1]):
            x = self._FracDiff(df.iloc[:,j], order=self.orders[j])
            fracDiffed.append( x )
        fracDiffed = pd.concat(fracDiffed, axis=1)
        return fracDiffed
    
    def InverseTransform(self, fracDiffed):
        """
        Inverts the fractional differentiation transformation. 
        Note that the full dataframe exactly as returned by `.transform()` is required,
        including any `NaN` padded missing values.
        
        Parameters
        ----------
            fracDiffed (pandas.DataFrame/np.array) Fractionally differentiated data
        """
        if not self.isFitted: 
            raise Exception('Fit the model first.')
        
        fracDiffed = pd.DataFrame(fracDiffed).sort_index()
        X = []
        for j in range(fracDiffed.shape[1]):
            memoryWeights = self._GetMemoryWeights(self.orders[j])
            K = len(memoryWeights)
            
            # Initial values
            lagData = self.lagData[j]
            lag_idx = lagData.index
            
            # Differenced values
            X_tilde = fracDiffed.iloc[:,j].dropna()
            idx = X_tilde.index
            
            # Iteratively invert transformation
            X_vals = np.ravel(self.lagData[j])
            X_tilde = np.ravel(X_tilde.values)
            for t in range(len(X_tilde)):
                x = -np.sum( memoryWeights[:-1] * X_vals[-(K-1):] ) + X_tilde[t]
                X_vals = np.append(X_vals, x)
            idx = np.concatenate( (lag_idx.values, idx.values) )
            X_vals = pd.Series(X_vals, index=idx)
            X.append( X_vals )
        X = pd.concat([ x for x in X ], axis=1)
        X.columns = fracDiffed.columns
        return X
    
    def _GetMemoryWeights(self, order, memoryThreshold=1e-4):
        """
        Returns an array of memory weights for each time lag.

        Parameters:
        -----------
            order           (float) Order of fracdiff
            memoryThreshold (float) Minimum magnitude of weight significance
        """
        memoryWeights = [1,]
        k = 1
        while True:
            weight = -memoryWeights[-1] * ( order - k + 1 ) / k # Iteratively generate next lag weight
            if abs(weight) < memoryThreshold:
                break
            memoryWeights.append(weight)
            k += 1
        return np.array(list(reversed(memoryWeights)))
    
    def _FracDiff(self, ts, order=1, memoryWeights=None):
        """
        Differentiates a time series based on a real-valued order.

        Parameters:
        -----------
            ts            (pandas.Series) Univariate time series
            order         (float) Order of differentiation
            memoryWeights (array) Optional pre-computed weights
        """
        if memoryWeights is None:
            memoryWeights = self._GetMemoryWeights(order)

        K = len(memoryWeights)
        fracDiffedSeries = ts.rolling(K).apply(lambda x: np.sum( x * memoryWeights ), raw=True)
        fracDiffedSeries = fracDiffedSeries.iloc[(K-1):]
        
        return fracDiffedSeries
    
    def _MinimumOrderSearch(self, ts, lowerOrder=0, upperOrder=1, first_run=False):
        """
        Binary search algorithm for estimating the minimum order of differentiation required for stationarity.
        
        Parameters
        ----------
            ts                   (pandas.Series) Univariate time series
            lowerOrder           (float) Lower bound on order
            upperOrder           (float) Upper bound on order
            first_run            (bool)  For testing endpoints of order bounds
        """  
        ## Convergence criteria
        if abs( upperOrder - lowerOrder ) <= self.precision:
            return upperOrder
        
        ## Initial run: Test endpoints
        if first_run:
            lowerFracDiff = self._FracDiff(ts, order=lowerOrder).dropna()
            upperFracDiff = self._FracDiff(ts, order=upperOrder).dropna()
            
            # Unit root tests
            lowerStationary = self.UnitRootTest.IsStationary( lowerFracDiff )
            upperStationary = self.UnitRootTest.IsStationary( upperFracDiff )

            # Series is I(0)
            if lowerStationary:
                return lowerOrder
            # Series is I(k>>1)
            if not upperStationary:                                                        
                print('Warning: Time series is explosive. Increase upper bounds.')
                return upperOrder
            
        ## Binary Search: Test midpoint
        midOrder = ( lowerOrder + upperOrder ) / 2                                      
        midFracDiff = self._FracDiff(ts, order=midOrder).dropna()
        midStationary = self.UnitRootTest.IsStationary( midFracDiff )
        
        # Series is weakly stationary in [lowerOrder, midOrder]
        if midStationary:
            return self._MinimumOrderSearch(ts, lowerOrder=lowerOrder, upperOrder=midOrder)
        # Series is weakly stationary in [midOrder, upperOrder]
        else:
            return self._MinimumOrderSearch(ts, lowerOrder=midOrder, upperOrder=upperOrder)
        