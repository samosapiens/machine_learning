#########################################################################################
# We use a Gaussian Naive Bayes model to predict if a stock will have a high return 
# or low return next Monday (num_holding_days = 5),  using as input decision variables 
#  the assets growthto yesterday from 2,3,,4,5,6,7,8,9 and 10 days before  
#########################################################################################

##################################################
# Imports
##################################################

# Pipeline and Quantopian Trading Functions
import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline, CustomFactor 
from quantopian.pipeline.factors import Returns, MACDSignal
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.classifiers.fundamentals import Sector as _Sector
from zipline.utils.numpy_utils import (
    repeat_first_axis,
    repeat_last_axis,
)
# The basics
from collections import OrderedDict
import time
import pandas as pd
import numpy as np

# SKLearn :)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

##################################################
# Globals
##################################################

num_holding_days = 5 # holding our stocks for five trading days.
days_for_fundamentals_analysis = 60
upper_percentile = 90
lower_percentile = 100 - upper_percentile
MAX_GROSS_EXPOSURE = 1.0
MAX_POSITION_CONCENTRATION = 0.01
MODEL = 'Logistic' # 'GaussianNB' # 'Logistic' # 'SVC'

class Sector(_Sector):
    window_safe = True

class MeanReversion1M(CustomFactor):
    inputs = (Returns(window_length=21),)
    window_length = 252
    def compute(self, today, assets, out, monthly_rets):
        np.divide(
            monthly_rets[-1] - np.nanmean(monthly_rets, axis=0),
            np.nanstd(monthly_rets, axis=0),
            out=out,
        )

class MoneyflowVolume5d(CustomFactor):
    inputs = (USEquityPricing.close, USEquityPricing.volume)
    # we need one more day to get the direction of the price on the first
    # day of our desired window of 5 days
    window_length = 6
    
    def compute(self, today, assets, out, close_extra, volume_extra):
        # slice off the extra row used to get the direction of the close
        # on the first day
        close = close_extra[1:]
        volume = volume_extra[1:]
        dollar_volume = close * volume
        denominator = dollar_volume.sum(axis=0)
        difference = np.diff(close_extra, axis=0)
        direction = np.where(difference > 0, 1, -1)
        numerator = (direction * dollar_volume).sum(axis=0)
        np.divide(numerator, denominator, out=out)

class PriceOscillator(CustomFactor):
    inputs = (USEquityPricing.close,)
    window_length = 252
    
    def compute(self, today, assets, out, close):
        four_week_period = close[-20:]
        np.divide(
            np.nanmean(four_week_period, axis=0),
            np.nanmean(close, axis=0),
            out=out,
        )
        out -= 1

class Trendline(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 252
    _x = np.arange(window_length)
    _x_var = np.var(_x)
    
    def compute(self, today, assets, out, close):
        x_matrix = repeat_last_axis(
            (self.window_length - 1) / 2 - self._x,
            len(assets),
        )
        y_bar = np.nanmean(close, axis=0)
        y_bars = repeat_first_axis(y_bar, self.window_length)
        y_matrix = close - y_bars
        np.divide(
            (x_matrix * y_matrix).sum(axis=0) / self._x_var,
            self.window_length,
            out=out,
        )

class Volatility3M(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 63
    
    def compute(self, today, assets, out, rets):
        np.nanstd(rets, axis=0, out=out)

class AdvancedMomentum(CustomFactor):
    inputs = (USEquityPricing.close, Returns(window_length=126))
    window_length = 252
    
    def compute(self, today, assets, out, prices, returns):
        np.divide(
            (
                (prices[-21] - prices[-252]) / prices[-252] -
                prices[-1] - prices[-21]
            ) / prices[-21],
            np.nanstd(returns, axis=0),
            out=out,
        )

asset_growth_3m = Returns(
    inputs=[Fundamentals.total_assets],
    window_length=63,
)

asset_to_equity_ratio = (
    Fundamentals.total_assets.latest /
    Fundamentals.common_stock_equity.latest
)

capex_to_cashflows = (
    Fundamentals.capital_expenditure.latest /
    Fundamentals.free_cash_flow.latest
)

ebitda_yield = (
    (Fundamentals.ebitda.latest * 4) /
    USEquityPricing.close.latest
)

ebita_to_assets = (
    (Fundamentals.ebit.latest * 4) /
    Fundamentals.total_assets.latest
)

return_on_total_invest_capital = Fundamentals.roic.latest
mean_reversion_1m = MeanReversion1M()
macd_signal_10d = MACDSignal(
    fast_period=12,
    slow_period=26,
    signal_period=10,
)

moneyflow_volume_5d = MoneyflowVolume5d()
net_income_margin = Fundamentals.net_margin.latest
operating_cashflows_to_assets = (
    (Fundamentals.operating_cash_flow.latest * 4) /
    Fundamentals.total_assets.latest
)

price_momentum_3m = Returns(window_length=63)
price_oscillator = PriceOscillator()
trendline = Trendline()
returns_39w = Returns(window_length=215)
volatility_3m = Volatility3M()
advanced_momentum = AdvancedMomentum()

features = {
    'Asset Growth 3M': asset_growth_3m,
    'Asset to Equity Ratio': asset_to_equity_ratio,
    'Capex to Cashflows': capex_to_cashflows,
    'EBIT to Assets': ebita_to_assets,
    'EBITDA Yield': ebitda_yield,
    'MACD Signal Line': macd_signal_10d,
    'Mean Reversion 1M': mean_reversion_1m,
    'Moneyflow Volume 5D': moneyflow_volume_5d,
    'Net Income Margin': net_income_margin,
    'Operating Cashflows to Assets': operating_cashflows_to_assets,
    'Price Momentum 3M': price_momentum_3m,
    'Price Oscillator': price_oscillator,
    'Return on Invest Capital': return_on_total_invest_capital,
    '39 Week Returns': returns_39w,
    'Trendline': trendline,
    'Volatility 3m': volatility_3m,
    'Advanced Momentum': advanced_momentum,
}


##################################################
# Initialize
##################################################

def initialize(context):
    """ Called once at the start of the algorithm. """

    # Configure the setup
#    set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))
#    set_asset_restrictions(security_lists.restrict_leveraged_etfs)

    # Schedule our function
    algo.schedule_function(
        rebalance,
        algo.date_rules.week_start(days_offset=0),
        algo.time_rules.market_open() 
    )

    # Build the Pipeline
    algo.attach_pipeline(make_pipeline(context), 'my_pipeline')

##################################################
# Pipeline-Related Code
##################################################
            
class Predictor(CustomFactor):
    """ Defines our machine learning model. """
    
    # The factors that we want to pass to the compute function. We use an ordered dict for clear labeling of our inputs.
    factor_dict = OrderedDict([
              ('Asset_Growth_2d' , Returns(window_length=2)),
              ('Asset_Growth_3d' , Returns(window_length=3)),
              ('Asset_Growth_4d' , Returns(window_length=4)),
              ('Asset_Growth_5d' , Returns(window_length=5)),
              ('Asset_Growth_6d' , Returns(window_length=6)),
              ('Asset_Growth_7d' , Returns(window_length=7)),
              ('Asset_Growth_8d' , Returns(window_length=8)),
              ('Asset_Growth_9d' , Returns(window_length=9)),
              ('Asset_Growth_10d' , Returns(window_length=10)),
              ('Return' , Returns(inputs=[USEquityPricing.open],window_length=5)),
              ('Asset Growth 3M', asset_growth_3m),
              ('Price Momentum 3M', price_momentum_3m),
              ('39 Week Returns', returns_39w)
              ])

    columns = factor_dict.keys()
    inputs = factor_dict.values()

    # Run it.
    def compute(self, today, assets, out, *inputs):
        """ Through trial and error, I determined that each item in the input array comes in with rows as days and securities as columns. Most recent data is at the "-1" index. Oldest is at 0.

        !!Note!! In the below code, I'm making the somewhat peculiar choice  of "stacking" the data... you don't have to do that... it's just a design choice... in most cases you'll probably implement this without stacking the data.
        """

        ## Import Data and define y.
        inputs = OrderedDict([(self.columns[i] , pd.DataFrame(inputs[i]).fillna(0,axis=1).fillna(0,axis=1)) for i in range(len(inputs))]) # bring in data with some null handling.
        num_secs = len(inputs['Return'].columns)
        y = inputs['Return'].shift(-num_holding_days-1)
        
        for index, row in y.iterrows():
            
             upper = np.nanpercentile(row, upper_percentile)
             lower = np.nanpercentile(row, lower_percentile)
             upper_mask = (row >= upper)
             lower_mask = (row <= lower)          
             row = np.zeros_like(row)
             row[upper_mask]= 1
             row[lower_mask]=-1
             y.iloc[index] = row
            
        y=y.stack(dropna=False)
        
        
        ## Get rid of our y value as an input into our machine learning algorithm.
        del inputs['Return']

        ## Munge x and y
        x = pd.concat([df.stack(dropna=False) for df in inputs.values()], axis=1).fillna(0)
        
        ## Run GaussianNB Model
        if MODEL == 'GaussianNB': 
            out[:] = self.GaussianNB_Model(x, y, num_secs)
        ## Run GaussianNB Model
        elif MODEL == 'Logistic':
            out[:] = self.Logistic_Model(x, y, num_secs)
        ## Run GaussianNB Model
        elif MODEL == 'SVC':
            out[:] = self.LinearSVC_Model(x, y, num_secs)
        ## Run GaussianNB Model by default
        else:
            out[:] = self.GaussianNB_Model(x, y, num_secs)
    
    def GaussianNB_Model(self, x, y, num_secs):
        model = GaussianNB()
        model_x = x[:-num_secs*(num_holding_days+1)]
        model_y = y[:-num_secs*(num_holding_days+1)]
        model.fit(model_x, model_y)
        return model.predict_proba(x[-num_secs:])[:, 1]
        
    def Logistic_Model(self, x, y, num_secs):
        model = LogisticRegression()
        model_x = x[:-num_secs*(num_holding_days+1)]
        model_y = y[:-num_secs*(num_holding_days+1)]
        model.fit(model_x, model_y)
        return model.predict_proba(x[-num_secs:])[:, 1]
        
    def LinearSVC_Model(self, x, y, num_secs):
        model = LinearSVC()
        model_x = x[:-num_secs*(num_holding_days+1)]
        model_y = y[:-num_secs*(num_holding_days+1)]
        model.fit(model_x, model_y)
        return model.decision_function(x[-num_secs:])

def make_pipeline(context):

    universe = QTradableStocksUS()
      
    predictions = Predictor(window_length=days_for_fundamentals_analysis, mask=universe)
    
    low_future_returns = predictions.percentile_between(0,lower_percentile)
    high_future_returns = predictions.percentile_between(upper_percentile,100)
   
    securities_to_trade = (low_future_returns | high_future_returns)
    pipe = Pipeline(
        columns={
            'predictions': predictions
        },
        screen=securities_to_trade
    )

    return pipe

def before_trading_start(context, data):
      
    context.output = algo.pipeline_output('my_pipeline')

    context.predictions = context.output['predictions']

##################################################
# Execution Functions
##################################################

def rebalance(context,data):
    # Timeit!
    start_time = time.time()
    
    objective = opt.MaximizeAlpha(context.predictions)
    
    max_gross_exposure = opt.MaxGrossExposure(MAX_GROSS_EXPOSURE)
    
    max_position_concentration = opt.PositionConcentration.with_equal_bounds(
        -MAX_POSITION_CONCENTRATION,
        MAX_POSITION_CONCENTRATION
    )
    
    dollar_neutral = opt.DollarNeutral()
    
    constraints = [
        max_gross_exposure,
        max_position_concentration,
        dollar_neutral,
    ]

    algo.order_optimal_portfolio(objective, constraints)

    # Print useful things. You could also track these with the "record" function.
    print 'Full Rebalance Computed Seconds: '+'{0:.2f}'.format(time.time() - start_time)
    print "Leverage: " + str(context.account.leverage)