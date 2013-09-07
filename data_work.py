# file: data_work.py
#
# "Growth under the Shadow of Expropriation"
#
# Mark Aguiar and Manuel Amador
#
# September 2010 (Updated August, 2012)
#
# This program uses the data in "data_www.csv" to generate the plots that are
# used in the paper. 
# 
# Python 2.7 code.

from __future__ import division
from csv import reader, QUOTE_NONNUMERIC
from scipy import log, array
from scipy.stats import linregress
from matplotlib import use
# use('Agg') # forces matplotlib to use a Non-GUI interface
from matplotlib import pyplot as plt

def myplot(x, y, cnames, xlim, ylim, xticks, yticks, xlabel, ylabel, 
           fname, fformat='pdf'):
    plt.plot(x, y, 'ko', markerfacecolor='k', markeredgewidth=1.5,
             markersize=5)
    for xi, yi, name in zip(x, y, cnames):
        plt.text(xi + 0.0005, yi, ' '+name, 
                 verticalalignment='baseline', alpha=0.7,
                 fontsize = 14)
    coef = linregress(x, y)
    plt.plot(xlim, array(xlim) * coef[0] + coef[1], 'k-', linewidth=3)
    plt.plot(xlim, [0, 0], '--', color='.3')
    plt.plot([0, 0], ylim, '--', color='.3')
    plt.xticks(xticks, ['$'+str(int(xi * 100)) + '\% $' for xi in xticks]) 
    plt.yticks(yticks, ['$'+str(int(yi * 100)) + '\% $' for yi in yticks])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
#    plt.savefig(fname, format=fformat)


rawdata = list(reader(open("data_www.csv"), quoting = QUOTE_NONNUMERIC))
# generating a dictionary with data of the relevant years
data={c: {z[0]:{zname:zvalue for zname,zvalue in zip(rawdata[0], z)} 
          for z in rawdata if z[rawdata[0].index('year')] == c} 
      for c in (2004, 1970)} 
gdpDeltaUS = log(data[2004]['United States']['per_cap_gdp']/
                 data[1970]['United States']['per_cap_gdp'])/34

GovChangeGDP = {'poor': [], 'rich': []}
GDPChange = {'poor': [], 'rich': []}
PrivChangeGDP  =  {'poor': [], 'rich': []}
Country = {'poor': [], 'rich': []}
NFAChangeGDP = {'poor': [], 'rich': []}
for c in data[1970].keys():
    try:
        d1, d0 = data[2004][c], data[1970][c]
        gdp_change = log(d1['per_cap_gdp']
                         /d0['per_cap_gdp'])/34 - gdpDeltaUS
        nfa_change_gdp = ((d1['assets']-d1['liabilities'])
                          /d1['dollar_gdp']
                          - (d0['assets']-d0['liabilities'])
                          /d0['dollar_gdp'])/34
        if (d0['per_cap_dollar'] < 10000): 
            gov_change_gdp = ((d1['reserves'] - d1['ppg_debt'])
                              /d1['dollar_gdp']
                              - (d0['reserves'] - d0['ppg_debt'])
                              /d0['dollar_gdp'])/34
            priv_change_gdp = ((d1['assets'] - d1['liabilities'] 
                                - d1['reserves'] + d1['ppg_debt'])
                               /d1['dollar_gdp']
                               - (d0['assets'] - d0['liabilities'] 
                                  - d0['reserves'] + d0['ppg_debt'])
                               /d0['dollar_gdp'])/34
            n = 'poor'
        else:
            n = 'rich'
    except:
        # don't add observation.
        pass
    else:
        NFAChangeGDP[n].append(nfa_change_gdp)
        GDPChange[n].append(gdp_change)
        Country[n].append(c)
        if n == 'poor':
            GovChangeGDP[n].append(gov_change_gdp) 
            PrivChangeGDP[n].append(priv_change_gdp)

plt.close('all')
plt.figure(1, figsize=[7,5])
myplot(GovChangeGDP['poor'], GDPChange['poor'], Country['poor'], 
       [-0.03, 0.015], [-0.04, 0.03], [-.02, -0.01, 0, 0.01], 
       [-.02, 0, .02], '$\Delta ({Govt}_{assets}/Y) / T$',  
       'Avg Growth Rate Relative to U.S.', 'plotI.pdf')
plt.figure(2, figsize=[7,5])
myplot(PrivChangeGDP['poor'], GDPChange['poor'], Country['poor'], 
       [-0.022, 0.03], [-0.04, 0.03], [-.02, -0.01, 0, 0.01], 
       [-.02, 0, .02], '$\Delta({Private}_{assets}/Y) / T$',
       'Avg Growth Rate Relative to U.S.', 'plotII.pdf')
plt.figure(3, figsize=[7,5])
myplot(NFAChangeGDP['rich'], GDPChange['rich'], Country['rich'], 
       [-0.022, 0.03], [-0.04, 0.03], [-.02, -0.01, 0, 0.01, 0.02], 
       [-.02, 0, .02], '$\Delta(NFA/Y) / T$',
       'Avg Growth Rate Relative to U.S.', 'plot_extra_1.pdf')
plt.figure(4, figsize=[7,5])
myplot(NFAChangeGDP['poor'], GDPChange['poor'], Country['poor'], 
       [-0.022, 0.03], [-0.04, 0.03], [-.02, -0.01, 0, 0.01, 0.02], 
       [-.02, 0, .02], '$\Delta(NFA/Y) / T$',
       'Avg Growth Rate Relative to U.S.', 'plot_extra_2.pdf')

plt.show()
