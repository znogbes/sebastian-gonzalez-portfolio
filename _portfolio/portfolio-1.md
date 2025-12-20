---
title: "Portfolio item number 1"
excerpt: "Short description of portfolio item number 1<br/><img src='/images/500x300.png'>"
collection: portfolio
---

This is an item in your portfolio. It can be have images or nice text. If you name the file .md, it will be parsed as markdown. If you name the file .html, it will be parsed as HTML. 

- Data used in accordance with Apache license 2.0
- provide info on environment


```python
# set up
import pandas as pd
import kaggle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

# upload data
try:
    # if already downloaded, just read csv
    data_train = pd.read_csv("data/train.csv")

except FileNotFoundError:
    # if downloading for the first time
    import os
    from kaggle.api.kaggle_api_extended import KaggleApi
    import zipfile

    # authenticate
    api = KaggleApi()
    api.authenticate()

    # download data
    # data directory
    data_dir = "./data/"
    #ensure directory exists
    os.makedirs(data_dir, exist_ok=True)

    api.competition_download_files(competition='playground-series-s5e11',
                                   path=data_dir)
    
    with zipfile.ZipFile("data/playground-series-s5e11.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
    
    data_train = pd.read_csv("data/train.csv")
```

- Explain data was generated synthetically, so don;t expect DQ issues, so just checking this is indeed the case 


```python
# data exploration (train data)
data_train.columns.to_list()
data_train.info() # no missing values in any columns 
# permanently remove id col
data_train.drop('id', inplace=True, axis=1)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 593994 entries, 0 to 593993
    Data columns (total 13 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   id                    593994 non-null  int64  
     1   annual_income         593994 non-null  float64
     2   debt_to_income_ratio  593994 non-null  float64
     3   credit_score          593994 non-null  int64  
     4   loan_amount           593994 non-null  float64
     5   interest_rate         593994 non-null  float64
     6   gender                593994 non-null  object 
     7   marital_status        593994 non-null  object 
     8   education_level       593994 non-null  object 
     9   employment_status     593994 non-null  object 
     10  loan_purpose          593994 non-null  object 
     11  grade_subgrade        593994 non-null  object 
     12  loan_paid_back        593994 non-null  float64
    dtypes: float64(5), int64(2), object(6)
    memory usage: 58.9+ MB
    


```python
# exploratory data analysis - numeric variables
# drop dependent variable (as it's been coded as 1 and 0)
data_train.describe().drop('loan_paid_back', axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>annual_income</th>
      <th>debt_to_income_ratio</th>
      <th>credit_score</th>
      <th>loan_amount</th>
      <th>interest_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>593994.000000</td>
      <td>593994.000000</td>
      <td>593994.000000</td>
      <td>593994.000000</td>
      <td>593994.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>48212.202976</td>
      <td>0.120696</td>
      <td>680.916009</td>
      <td>15020.297629</td>
      <td>12.356345</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26711.942078</td>
      <td>0.068573</td>
      <td>55.424956</td>
      <td>6926.530568</td>
      <td>2.008959</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6002.430000</td>
      <td>0.011000</td>
      <td>395.000000</td>
      <td>500.090000</td>
      <td>3.200000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27934.400000</td>
      <td>0.072000</td>
      <td>646.000000</td>
      <td>10279.620000</td>
      <td>10.990000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>46557.680000</td>
      <td>0.096000</td>
      <td>682.000000</td>
      <td>15000.220000</td>
      <td>12.370000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60981.320000</td>
      <td>0.156000</td>
      <td>719.000000</td>
      <td>18858.580000</td>
      <td>13.680000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>393381.740000</td>
      <td>0.627000</td>
      <td>849.000000</td>
      <td>48959.950000</td>
      <td>20.990000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create df's by payback group 
filter = data_train['loan_paid_back'] == 1
data_train_payers = data_train[filter]
filter = data_train['loan_paid_back'] == 0
data_train_non_payers = data_train[filter]

# visualisation of numeric variables
# define cols to loop through in plot
numeric_cols = data_train_payers.select_dtypes(
    include=np.number).columns.drop('loan_paid_back')
# determine grid size
num_cols = len(numeric_cols)
ncols = 2
nrows = int(np.ceil(num_cols/2)) # in case of single row/col

# define plot objects
fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows*2))
axes = axes.flatten()

# make boxplots
for i, col in enumerate(numeric_cols):
    ax = axes[i]

    # create boxplot for col
    bp = ax.boxplot(
        [data_train_payers[col], data_train_non_payers[col]],
        labels=['Payers', 'Non-Payers'],
        # allow custom colours 
        patch_artist=True,
        # horizontal orientation of bars
        vert=False,
        # make whiskers represent min and max values
        whis = (0, 100)
        )
    
    colors = ['blue', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bp['medians']:
        median.set(color='black')
    
    ax.set_title(f"Distribution of {col} by outcome group")

#hide unused subplots
for i in range(num_cols, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()
```


    
![png](intro-and-eda_files/intro-and-eda_5_0.png)


<details markdown="1">
<summary>Explain boxplots vs histograms (depending on audience)</summary>

```python
# alternatively, these variables can also be visualised as histograms
fig, axes = plt.subplots(nrows,ncols, figsize=(12, nrows*2))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]

    # create histograms for each group
    ax.hist(
        data_train_payers[col], bins=50,  
        color='blue',
        label='Payers', 
        histtype = 'step'
        )
        
    ax.hist(
        data_train_non_payers[col], bins=20, 
        color='red',
        label='Non-payers', histtype = 'step'
        )
    
    ax.set_title(f"Distribution of {col} by outcome group")
    ax.legend()

#hide unused subplots
for i in range(num_cols, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()
```

    
![png](intro-and-eda_files/intro-and-eda_7_0.png)

</details>

- Visualisation of categorical variables
- Explain functions


```python
# visualisation of categorical variables

# recode outcome variable
recode_dict = {1:'Payers', 0:'Non-Payers'}
data_train['loan_paid_back_recoded'] =  data_train['loan_paid_back'].map(recode_dict)

# generate frequency tables by outcome group
def generate_freq_table(data, col_name):
    # frequencies of categorical vol by outcome 
    count_table = pd.crosstab(data[col_name], data['loan_paid_back_recoded'])
    # reset index so categorical variable is not kept as index
    count_table = count_table.reset_index()
    # calculate percentages by outcome group
    percent_payers = (
        count_table['Payers']/count_table['Payers'].sum()*100).round(1)
    percent_non_payers = (
        count_table['Non-Payers']/count_table['Non-Payers'].sum()*100).round(1)
    count_table['percent_payers'] = percent_payers
    count_table['percent_non_payers'] = percent_non_payers
    # sort values by count of payers 
    frequency_table = count_table.sort_values(by='Payers', ascending=False)
    return frequency_table

# function to plot frequencies by group
def plot_categorical_cols(freq_table, col_name):
    # drop count columns, as the plot is for percentages
    freq_table_tidy = freq_table.drop(['Payers', 'Non-Payers'], axis=1)
    freq_table_tidy.plot(x = col_name, kind='barh', color=['blue', 'red'])
    # axis labels
    plt.xlabel('Percent')
    plt.gca().xaxis.set_major_formatter(
        # data already scalled to percent, 100
        PercentFormatter(100))
    plt.ylabel('')
    # show bars in desc order
    plt.gca().invert_yaxis()
    plt.title(f"Breakdown of outcome group by \n{col_name}")
    custom_labels = ['Payers', 'Non-Payers']
    plt.legend(loc='lower right', frameon = False, labels=custom_labels)
    plot = plt.show()
    return plot
```

- Deploy function via loop


```python
# loop over to see freq tables and plots
categorical_cols = data_train.select_dtypes(include='object').drop(
    'loan_paid_back_recoded', axis=1)

for col in categorical_cols:
    print(f"\nBreakdown of outcome group by \n{col}")
    freq_table = generate_freq_table(data_train, col)
    print(freq_table)
    plot = plot_categorical_cols(freq_table, col)
    print(plot)
```

    
    Breakdown of outcome group by 
    gender
    loan_paid_back_recoded  gender  Non-Payers  Payers  percent_payers  \
    0                       Female       60712  245463            51.7   
    1                         Male       58025  226066            47.6   
    2                        Other         763    2965             0.6   
    
    loan_paid_back_recoded  percent_non_payers  
    0                                     50.8  
    1                                     48.6  
    2                                      0.6  
    


    
![png](intro-and-eda_files/intro-and-eda_11_1.png)
    


    None
    
    Breakdown of outcome group by 
    marital_status
    loan_paid_back_recoded marital_status  Non-Payers  Payers  percent_payers  \
    2                              Single       58094  230749            48.6   
    1                             Married       55685  221554            46.7   
    0                            Divorced        4334   16978             3.6   
    3                             Widowed        1387    5213             1.1   
    
    loan_paid_back_recoded  percent_non_payers  
    2                                     48.6  
    1                                     46.6  
    0                                      3.6  
    3                                      1.2  
    


    
![png](intro-and-eda_files/intro-and-eda_11_3.png)
    


    None
    
    Breakdown of outcome group by 
    education_level
    loan_paid_back_recoded education_level  Non-Payers  Payers  percent_payers  \
    0                           Bachelor's       59027  220579            46.5   
    1                          High School       34938  148654            31.3   
    2                             Master's       18401   74696            15.7   
    3                                Other        5261   21416             4.5   
    4                                  PhD        1873    9149             1.9   
    
    loan_paid_back_recoded  percent_non_payers  
    0                                     49.4  
    1                                     29.2  
    2                                     15.4  
    3                                      4.4  
    4                                      1.6  
    


    
![png](intro-and-eda_files/intro-and-eda_11_5.png)
    


    None
    
    Breakdown of outcome group by 
    employment_status
    loan_paid_back_recoded employment_status  Non-Payers  Payers  percent_payers  \
    0                               Employed       47703  402942            84.9   
    2                          Self-employed        5329   47151             9.9   
    1                                Retired          46   16407             3.5   
    4                             Unemployed       57635    4850             1.0   
    3                                Student        8787    3144             0.7   
    
    loan_paid_back_recoded  percent_non_payers  
    0                                     39.9  
    2                                      4.5  
    1                                      0.0  
    4                                     48.2  
    3                                      7.4  
    


    
![png](intro-and-eda_files/intro-and-eda_11_7.png)
    


    None
    
    Breakdown of outcome group by 
    loan_purpose
    loan_paid_back_recoded        loan_purpose  Non-Payers  Payers  \
    2                       Debt consolidation       65942  258753   
    6                                    Other       12623   51251   
    1                                      Car       11585   46523   
    4                                     Home        7799   36319   
    0                                 Business        6598   28705   
    3                                Education        8169   28472   
    5                                  Medical        5061   17745   
    7                                 Vacation        1723    6726   
    
    loan_paid_back_recoded  percent_payers  percent_non_payers  
    2                                 54.5                55.2  
    6                                 10.8                10.6  
    1                                  9.8                 9.7  
    4                                  7.7                 6.5  
    0                                  6.0                 5.5  
    3                                  6.0                 6.8  
    5                                  3.7                 4.2  
    7                                  1.4                 1.4  
    


    
![png](intro-and-eda_files/intro-and-eda_11_9.png)
    


    None
    
    Breakdown of outcome group by 
    grade_subgrade
    loan_paid_back_recoded grade_subgrade  Non-Payers  Payers  percent_payers  \
    12                                 C3        9626   49069            10.3   
    13                                 C4        8730   47227            10.0   
    11                                 C2        8103   46340             9.8   
    10                                 C1        7466   45897             9.7   
    14                                 C5        8197   45120             9.5   
    15                                 D1        9928   27101             5.7   
    17                                 D3       11156   25538             5.4   
    18                                 D4       10012   25085             5.3   
    16                                 D2        9608   24824             5.2   
    19                                 D5        9213   22888             4.8   
    6                                  B2         949   14218             3.0   
    5                                  B1        1200   13144             2.8   
    7                                  B3         835   13091             2.8   
    9                                  B5         917   13020             2.7   
    8                                  B4         947   12930             2.7   
    23                                 E4        2816    5220             1.1   
    22                                 E3        2534    4541             1.0   
    20                                 E1        2398    4493             0.9   
    21                                 E2        2149    4223             0.9   
    24                                 E5        2011    4073             0.9   
    29                                 F5        2145    3802             0.8   
    28                                 F4        2009    3526             0.7   
    25                                 F1        2078    3456             0.7   
    26                                 F2        1989    3214             0.7   
    27                                 F3        2012    3070             0.6   
    4                                  A5         136    2335             0.5   
    2                                  A3          92    1974             0.4   
    1                                  A2          95    1923             0.4   
    3                                  A4          73    1628             0.3   
    0                                  A1          76    1524             0.3   
    
    loan_paid_back_recoded  percent_non_payers  
    12                                     8.1  
    13                                     7.3  
    11                                     6.8  
    10                                     6.2  
    14                                     6.9  
    15                                     8.3  
    17                                     9.3  
    18                                     8.4  
    16                                     8.0  
    19                                     7.7  
    6                                      0.8  
    5                                      1.0  
    7                                      0.7  
    9                                      0.8  
    8                                      0.8  
    23                                     2.4  
    22                                     2.1  
    20                                     2.0  
    21                                     1.8  
    24                                     1.7  
    29                                     1.8  
    28                                     1.7  
    25                                     1.7  
    26                                     1.7  
    27                                     1.7  
    4                                      0.1  
    2                                      0.1  
    1                                      0.1  
    3                                      0.1  
    0                                      0.1  
    


    
![png](intro-and-eda_files/intro-and-eda_11_11.png)
    


    None
    

- What we've learned / can expect in modelling

Next: data pre-processing
