import arch
from arch.unitroot import PhillipsPerron as PP
from arch.unitroot import ADF

## TODO: Ng and Perron (2001)?

class PhillipsPerron:
    """
    Unit root testing via Phillips and Perron (1988). This test is robust to
    serial correlation and heteroskedasticity.

    References:
    -----------
    Phillips, P. C. B., & Perron, P. (1988). Testing for a unit root in time series regression. 
    Biometrika, 75(2), 335–346. https://doi.org/10.1093/biomet/75.2.335
    """
    
    def __init__(self, 
                config={ 'trend' : 'n', 'test_type' : 'tau'}, 
                significance=0.01,
                checkCV=False, 
                cv_sig=None):
        self.config = config
        self.significance = significance
        self.checkCV = checkCV
        self.cv_sig = cv_sig

    def IsStationary(self, ts):
        """
        Performs a unit root test.
        """

        testResults = PP(ts, trend=self.config['trend'], test_type=self.config['test_type'])
        pval, cv, stat = testResults.pvalue, testResults.critical_values, testResults.stat

        result = self.HypothesisTest(pval, cv, stat)

        return result

    def HypothesisTest(self, pval, cv, stat):
        """
        Null Hypothesis: Time series is integrated of order I(1)
        Alt Hypothesis: Time series is integrated of order I(k<1)
        """
        
        # Reject the hypothesis
        if (pval < self.significance) or ( self.checkCV and (stat < cv.get(self.cv_sig, 0)) ):
            return True
        # Fail to reject the hypothesis
        else:
            return False

class ADFuller:
    """
    Unit root testing via Said and Dickey (1984). This test assumes a parametric
    ARMA structure to correct for serial correlation but assumes the errors are homoskedastic.

    References:
    -----------
    Said E. Said, & Dickey, D. A. (1984). Testing for Unit Roots in Autoregressive-Moving Average 
    Models of Unknown Order. Biometrika, 71(3), 599–607. https://doi.org/10.2307/2336570
    """
    def __init__(self, 
                config={ 'trend' : 'n', 'method' : 'AIC'}, 
                significance=0.01,
                checkCV=False, 
                cv_sig=None):
        self.config = config
        self.significance = significance
        self.checkCV = checkCV
        self.cv_sig = cv_sig

        ## Compatability workaround //
        #   arch <= 4.17 uses capital letters but newer versions use lowercase
        if (str(arch.__version__) > '4.17'):
            if self.config.get('method') == 'AIC':
                self.config['method'] = 'aic'
            elif self.config.get('method') == 'BIC':
                self.config['method'] = 'bic'

    def IsStationary(self, ts):
        """
        Performs a unit root test.
        """

        testResults = ADF(ts, trend=self.config['trend'], method=self.config['method'])
        pval, cv, stat = testResults.pvalue, testResults.critical_values, testResults.stat

        result = self.HypothesisTest(pval, cv, stat)

        return result

    def HypothesisTest(self, pval, cv, stat):
        """
        Null Hypothesis: Gamma = 0 (Unit root)
        Alt Hypothesis: Gamma < 0
        """
        
        # Reject the hypothesis
        if (pval < self.significance) or ( self.checkCV and (stat < cv.get(self.cv_sig, 0)) ):
            return True
        # Fail to reject the hypothesis
        else:
            return False

    

    