import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt 
import seaborn as sns

datafile = './data/'
data_train = pd.read_csv(datafile +'train.csv')
data_test = pd.read_csv(datafile + 'test.csv')

"""
数据说明

SeriousDlqin2yrs：违约客户及超过90天逾期客户，bool型；
RevolvingUtilizationOfUnsecuredLines：贷款以及信用卡可用额度与总额度比例，百分比；
age：用户年龄，整型
NumberOfTime30-59DaysPastDueNotWorse：35-59天逾期但不糟糕次数，整型；
DebtRatio：负债率，百分比；
MonthlyIncome：月收入，整型；
NumberOfOpenCreditLinesAndLoans：开放式信贷和贷款数量，开放式贷款（分期付款如汽车贷款或抵押贷款）和信贷（如信用卡）的数量，整型；
NumberOfTimes90DaysLate：90天逾期次数：借款者有90天或更高逾期的次数，整型；
NumberRealEstateLoansOrLines：不动产贷款或额度数量：抵押贷款和不动产放款包括房屋净值信贷额度，整型；
NumberOfTime60-89DaysPastDueNotWorse：60-89天逾期但不糟糕次数：借款人在在过去两年内有60-89天逾期还款但不糟糕的次数，整型；
NumberOfDependents：家属数量：不包括本人在内的家属数量，整型；
"""
