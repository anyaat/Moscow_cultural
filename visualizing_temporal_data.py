import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# если в ipython notebook, раскоментируйте следующую строчку
# %matplotlib inline

def get_months(yearS, yearE):
    """
    В соответствии с заданным временным промежутком создаем списки дат и названий месяцев. 
    """
    monthNames = []
    if yearE != 2016:
        for year in range(yearS, yearE + 1):
            mN = []
            year = str(year)
            for month in 'jan feb mar apr may jun jul aug sep oct nov dec'.split():
                mN.append(month + year[-2:])
            monthNames.append(mN)
        months = pd.date_range(str(yearS-1)+'-12-31', str(yearE)+'-12-28', freq = 'M').shift(15,
                                                                                freq=pd.datetools.day)
    else:
        for year in range(yearS, yearE):
            mN = []
            year = str(year)
            for month in 'jan feb mar apr may jun jul aug sep oct nov dec'.split():
                mN.append(month + year[-2:])
            monthNames.append(mN)
        mN = []
        for month in 'jan feb mar apr may'.split():
            mN.append(month + '16')
        monthNames.append(mN)
        months = pd.date_range(str(yearS-1)+'-12-31', '2016-05-28', freq = 'M').shift(15,
                                                                                freq=pd.datetools.day)
    return monthNames, months

def add_dates_count(yearS, yearE, df):
    """
    Для каждого события считаем сколько оно произошла в каждом месяце заданного интервала.
    Создаем копию исходного data frame и добавляем в новый data frame полученные значения.  
    """
    allMonths = []
    monthNames, months = get_months(yearS, yearE)
    
    # считаем, в каком месяце сколько раз состоялось мероприятие
    for year, mN in zip(range(yearS, yearE + 1), monthNames):
        year = str(year)
        aM = [] 
        for m in range(len(mN)):
            aM.append([])
        r = len(mN) + 1
        for row in df.iterrows():
            for i, month in zip(range(1, r), aM):
                if str(i) + '.' + year in row[1]['date']:
                    count = 0
                    for day in row[1]['date'].split():
                        if str(i) + '.' + year in day:
                            count += 1
                    month.append(count)
                else:
                    month.append(np.nan)
        allMonths.append(aM)
                   
    # добавляем новые столбцы в исходный data frame
    dfExt = df
    monthNames = sum(monthNames, [])
    allMonths = sum(allMonths, [])
    for name, month in zip(monthNames, allMonths):
        dfExt[name] = month
        
    return dfExt, monthNames, months

def get_dates_category(yearS, yearE, df, normalizeBYmonth=False):
    """
    Создаем data frame c количеством событий каждой категории в каждом месяце указанного промежутка.  
    """
    categoryBYmonth = []
    dfExt, monthNames, months = add_dates_count(yearS, yearE, df)
        
    # считаем, сколько в каком месяце прошло мероприятий каждой из категорий
    for month in monthNames:
        categoryBYmonth.append(dfExt.ix[:,['category',month]].dropna().groupby('category').aggregate(sum)[month])
        
    # записываем все в dataframe
    dateCATEGORY = pd.DataFrame(categoryBYmonth, index=months, columns=['выставки', 'кинопоказы', 
                                                        'концерты', 'лекции мк', 'спектакли', 'фестивали'])
    dateCATEGORY = dateCATEGORY.fillna(0)
    dateCATEGORY.columns = ['exhibitions', 'films', 'concerts', 'lectures', 'plays', 'festivals']
    
    if normalizeBYmonth == True:
        # нормализуем по долe среди других категорий в данном месяце
        dateCATEGORY['sum'] = dateCATEGORY.sum(axis=1)
        dateCATEGORY.loc[:,'exhibitions':'festivals'] = \
        dateCATEGORY.loc[:,'exhibitions':'festivals'].div(dateCATEGORY['sum'], axis=0)
        dateCATEGORY = dateCATEGORY.drop('sum', axis=1)
        
    return dateCATEGORY

def get_dates_price(yearS, yearE, df, normalizeBYmonth=False):
    """
    Создаем data frame c количеством событий каждой ценовой категории в каждом месяце указанного промежутка.  
    """
    price_gr = []
    priceBYmonth = []
    dfExt, monthNames, months = add_dates_count(yearS, yearE, df)
    
    # добавляем новый столбец с ценовыми категориями
    for row in df['price']:
        if row == 'free':
            price_gr.append('free')
        elif float(row) <= 100:
            price_gr.append('< 100')
        elif float(row) <= 500:
            price_gr.append('< 500')
        elif float(row) <= 1000:
            price_gr.append('< 1000')
        elif float(row) <= 5000:
            price_gr.append('< 5000')
        elif float(row) <= 64000:
            price_gr.append('< 63000')
        else:
            price_gr.append(row)
    dfExt['price_gr'] = price_gr
        
    # считаем, сколько в каком месяце прошло мероприятий каждой ценовой категории 
    for month in monthNames:
        priceBYmonth.append(dfExt.ix[:,[month,'price_gr']].dropna().groupby('price_gr').aggregate(sum)[month])
        
    # записываем все в data frame
    datePRICE = pd.DataFrame(priceBYmonth, index=months, columns=['< 100', '< 500', '< 1000', '< 5000', 
                                                                  '< 63000', 'free'])
    datePRICE = datePRICE.fillna(0)
    
    if normalizeBYmonth == True:
        datePRICE['sum'] = datePRICE.sum(axis=1)
        datePRICE.loc[:,'< 100':'free'] = \
        datePRICE.loc[:,'< 100':'free'].div(datePRICE['sum'], axis=0)
        datePRICE = datePRICE.drop('sum', axis=1)
        
    return datePRICE

def normalize(dateVARIABLE):
    """
    Нормализуем значения data frame по категориальной переменной.
    """
    dateVARIABLE.loc['sum'] = dateVARIABLE.sum(axis=0)
    dateVARIABLE.loc[list(dateVARIABLE.index)[:-1]] = \
            dateVARIABLE.loc[list(dateVARIABLE.index)[:-1]].div(dateVARIABLE.loc['sum'], axis=1)
    dateVARIABLE = dateVARIABLE.drop('sum', axis=0)
    return dateVARIABLE

def visualize(yearS, yearE, df, variable):
    """
    Строим графики распределения заданной категориальной переменной по месяцам.
    """
    vars('get_dates_'+variable)(yearS, yearE, df).plot(linewidth=2.0)
    vars('get_dates_'+variable)(yearS, yearE, df, True).plot(linewidth=2.0)
    normalize(vars('get_dates_'+variable)(yearS, yearE, df)).plot(linewidth=2.0)

def main():
    matplotlib.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (17, 7)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12

    mc = pd.read_csv('moscult.csv', sep='\t')
    variables = 'category price'.split()
    for var in variables:
        visualize(2014, 2016, mc, var)    

if __name__ == "__main__":
    main()

