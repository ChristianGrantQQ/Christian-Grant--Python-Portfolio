# Christian-Grant--Python-Portfolio
For a class "final project"
BISC 450C




Importing Data Packages

%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set(style = "darkgrid")
Reading Imported File

df = pd.read_csv('/home/student/Desktop/classroom/myfiles/Notebooks/fortune500.csv')
Reading Head of File

df.head()
Year	Rank	Company	Revenue (in millions)	Profit (In millions)
0	1995	1	General Motors	9823.5	806
1	1995	2	Exxon Mobil	5661.4	584.8
2	1995	3	U.S. Steel	3250.4	195.4
3	1995	4	General Electric	2959.1	212.6
4	1995	5	Esmark	2510.8	19.1
Reading Tail of File

df.tail()
Year	Rank	Company	Revenue (in millions)	Profit (In millions)
25495	2005	496	Wm. Wrigley Jr.	3648.6	493
25496	2005	497	Peabody Energy	3631.6	175.4
25497	2005	498	Wendy's International	3630.4	57.8
25498	2005	499	Kindred Healthcare	3616.6	70.6
25499	2005	500	Cincinnati Financial	3614.0	584
Renaming Columns of File

df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
year	rank	company	revenue	profit
0	1995	1	General Motors	9823.5	806
1	1995	2	Exxon Mobil	5661.4	584.8
2	1995	3	U.S. Steel	3250.4	195.4
3	1995	4	General Electric	2959.1	212.6
4	1995	5	Esmark	2510.8	19.1
Figuring out Ammount of Rows

len(df)
25500
Listing Types of Various Columns

df.dtypes
year         int64
rank         int64
company     object
revenue    float64
profit      object
dtype: object
Listing Non-Numeric Profit Items Head

non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head()
year	rank	company	revenue	profit
228	1955	229	Norton	135.0	N.A.
290	1955	291	Schlitz Brewing	100.0	N.A.
294	1955	295	Pacific Vegetable Oil	97.9	N.A.
296	1955	297	Liebmann Breweries	96.0	N.A.
352	1955	353	Minneapolis-Moline	77.4	N.A.
Listing All Non-Numeric Profit Sets

set(df.profit[non_numeric_profits])
{'N.A.'}
Listing How Many N.A. Values Exist in Data Set

len(df.profit[non_numeric_profits])
369
Plotting Histiogram of N.A. Values by Year

bin_sizes, _, _ = plt.hist(df.year[non_numeric_profits], bins= range(1955, 2006) )
Histiogram

Saving Over Data Frame and Removing Non-Numeric Profits

df = df.loc[~non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
Calculating New Length

len(df)
25131 
Checking If the N.A. was properly removed

df.dtypes
year         int64
rank         int64
company     object
revenue    float64
profit     float64
dtype: object
Making a Plot of 'Increase in mean Fortune 500 company profits from 1995 to 2005'

group_by_year = df.loc[:, ['year','revenue','profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x = 0,y = 0 )
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1995 to 2005', 'Profit (millions)')
Yet Another Plot

Ploting by Revenue Instead

y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue (Millions)')
Need to check if this is working soon

Making an Overlay Plot with Standard Error of Both Revenue and Profit Graphs

def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha = 0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols = 2)
title = 'Increase in mean and std fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year. std().profit.values
stds2 = group_by_year. std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'Profits', 'Profit(millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'Revenue', 'Revenue (Millions)')
fig. set_size_inches(14,4)
fig.tight_layout()
Done
