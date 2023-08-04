import pandas as pd
import numpy as np

class dataProduct:
    #creating the dataProduct class with all the variables
    def __init__(self,file_name,price,cost,deltaPrice,tax):
        df = pd.read_csv(file_name)
        results = Initialize(df)
        self.n = results[0]
        self.T = results[1]
        self.numStore = results[2]
        self.sales = results[3]
        self.date = results[4]
        self.stock = results[5]
        self.state = results[6]
        self.region = results[7]
        self.TreatGroup = results[8]
        self.T0  = results[9]
        self.ExogenousCovariates = results[10]
        self.salesTreat = results[11]
        self.salesControl = results[12]
        self.numTreat = results[13]
        self.numControl = results[14]
        self.ExogenousCovariatesOOS = results[15]
        self.salesTreatOOS = results[16]
        self.salesControlOOS= results[17]
        self.stateSet= results[18]
        self.numState= results[19]
        self.numStoreStateTreat = results[20]
        self.numStoreStateControl = results[21]
        self.salesTreatState= results[22]
        self.salesControlState = results[23]
        self.salesTreatStateOOS= results[24]
        self.salesControlStateOOS= results[25]
        self.price = price
        self.cost = cost
        self.deltaPrice = deltaPrice
        self.tax = tax

#initializing values to variables
def Initialize(df):
        N = df.shape[0] #number of units
        df['Date'] = pd.to_datetime(df['Date'])
        dates_full = df['Date']
        data_date = np.array(dates_full.to_list())
        day = df['Date'].dt.day
        data_day = np.array(day.to_list())
        month = df['Date'].dt.month
        data_month = np.array(month.to_list())
        year = df['Date'].dt.year
        data_year = np.array(year.to_list())

        n = np.where(np.diff(data_day)>0)[0][0] + 1
        T = int(N/n) #time period
        

        VDA_QTD = np.array(df['VDA_QTD'].to_list())
        Q_list = np.reshape(VDA_QTD,(T,n))
        Q = pd.DataFrame(Q_list) #sales

        nS = np.array(df['Opened'].to_list())
        numStoreList = np.reshape(nS,(T,n))
        numStore = pd.DataFrame(numStoreList)

        idx_zero = Q.columns[Q.std() == 0].to_list() #finding where the standard deviation is 0 and removing those columns
  
        Q = Q.drop(columns = idx_zero)
        numStore = numStore.drop(columns=idx_zero)
        maxStore = numStore.max()
        Y = Q/maxStore
        sales = Y

        date_list = np.reshape(np.array(data_date),(T,n))
        date  = pd.DataFrame(date_list)
        date = date.drop(columns = idx_zero)
        date = date[0]

        sl = np.array(df['EST_QTD'].to_list())
        stockList = np.reshape(sl,(T,n))
        stock = pd.DataFrame(stockList)
        stock = stock.drop(columns = idx_zero)

        st = np.array(df['State'].to_list())
        stateList = np.reshape(st,(T,n))
        state = pd.DataFrame(stateList)
        state = state.drop(columns = idx_zero) #organizing state, region and stock data

        re = np.array(df['Region'].to_list())
        reList = np.reshape(re,(T,n))
        region = pd.DataFrame(reList)
        region = region.drop(columns = idx_zero)

        gt = np.array(df['Treatment Group'].to_list())
        gtList = np.reshape(gt,(T,n))
        gtreat = pd.DataFrame(gtList)
        gtreat = gtreat.drop(columns = idx_zero)
        rows_to_delete = list(range(1,T))
        gtreat = gtreat.drop(rows_to_delete) #finding treatment groups
        TreatGroup = gtreat

        tr = np.array(df['Period'].to_list())
        trList = np.reshape(tr,(T,n))
        treat = pd.DataFrame(trList)
        columns_to_delete = list(range(1,n))
        columns_to_delete.append(idx_zero[0])
        set_list = set(columns_to_delete)
        columns_to_delete = list(set_list)
        treat = treat.drop(columns = columns_to_delete) 

        T0 = treat.ne(0).idxmax()[0]
        w = df['Date'].dt.day_name()
        wdlist = np.reshape(np.array(w.to_list()),(T,n))
        wd = pd.DataFrame(wdlist)
        columns_to_delete = list(range(1,n))
        wd = wd.drop(columns= columns_to_delete)
        wd = pd.get_dummies(wd)

        trend = list(range(0,T))

        mat1 = wd
        rows_to_delete = list(range(T0,T))
        mat1 = mat1.drop(rows_to_delete)
        mat1 = mat1.drop(columns = "0_Sunday")
        mat2 = pd.DataFrame(trend)
        mat2 = mat2.drop(rows_to_delete)
        result = mat2
        l = ["0_Friday","0_Monday","0_Tuesday","0_Wednesday","0_Thursday","0_Saturday"]
        extracted_columns = mat1[l]
        result = result.join(extracted_columns)
        ExogenousCovariates = result
        desired_order = ["0_Monday","0_Tuesday","0_Wednesday","0_Thursday","0_Friday","0_Saturday"]
        ExogenousCovariates = ExogenousCovariates[desired_order] #finding exogenous covariates with respect to weekdays, we drop sunday as it is an outlier



        n = Y.shape[1]
        condition = (gtreat == True).any()
        columns_treat = gtreat.columns[gtreat.iloc[0] == True].to_list()
        salesTreat = Y
        length = salesTreat.shape[0]
        rows_to_delete = list(range(T0,length))
        salesTreat = salesTreat.drop(rows_to_delete)
        salesTreat = salesTreat[columns_treat]

        columns_control = gtreat.columns[gtreat.iloc[0] == False].to_list()
        salesControl = Y
        rows_to_delete = list(range(T0,length))
        salesControl = salesControl.drop(rows_to_delete)
        salesControl = salesControl[columns_control] #sales of treatment and control groups before interventionn

        numTreat = salesTreat.shape[1]
        numControl = salesControl.shape[1]

        mat1 = wd
        rows_to_delete = list(range(0,T0))
        mat1 = mat1.drop(rows_to_delete)
        mat1 = mat1.drop(columns = "0_Sunday")
        mat2 = pd.DataFrame(trend)
        mat2 = mat2.drop(rows_to_delete)
        result = mat2
        l = ["0_Friday","0_Monday","0_Tuesday","0_Wednesday","0_Thursday","0_Saturday"]
        extracted_columns = mat1[l]
        result = result.join(extracted_columns)
        ExogenousCovariatesOOS = result
        desired_order = ["0_Monday","0_Tuesday","0_Wednesday","0_Thursday","0_Friday","0_Saturday"]
        ExogenousCovariatesOOS = ExogenousCovariatesOOS[desired_order] #exogenous covariates after intervention


        condition = (gtreat == True).any()
        columns_treat = gtreat.columns[gtreat.iloc[0] == True].to_list()
        salesTreatOOS= Y
        rows_to_delete = list(range(0,T0))
        salesTreatOOS = salesTreatOOS.drop(rows_to_delete)
        salesTreatOOS = salesTreatOOS[columns_treat] 

        columns_control = gtreat.columns[gtreat.iloc[0] == False].to_list()
        salesControlOOS = Y
        rows_to_delete = list(range(0,T0))
        salesControlOOS = salesControlOOS.drop(rows_to_delete)
        salesControlOOS = salesControlOOS[columns_control] #sales of treatment and control groups after intervention

        stateSet = list(set(state.values[0])) #list of all states
        numState = len(stateSet)


        stateSet.sort()

        QS_treat = np.empty((T,numState))
        QS_control = np.empty((T,numState))
        numStoreStateTreat = np.ones((T,numState))
        numStoreStateControl = np.ones((T,numState)) 
        #organizing data at a state level
        for i in range(0,len(stateSet)):

            state_name = stateSet[i]

            state_columns = state.columns[state.iloc[0] == state_name].to_list()
            columns_treat = gtreat.columns[gtreat.iloc[0] == True].to_list()
            columns_control = gtreat.columns[gtreat.iloc[0] == False].to_list()

            state_treat_columns = [value for value in state_columns if value in columns_treat]
            state_control_columns = [value for value in state_columns if value in columns_control]

            Q_sub_treat = Q[state_treat_columns]
            
            sum_row = Q_sub_treat.sum(axis = 1)
            ms_rows = maxStore[state_treat_columns]
            ms_sum = sum(ms_rows)

            QS_treat[:,i] = sum_row/ms_sum
    
            Q_sub_control = Q[state_control_columns]
            sum_row = Q_sub_control.sum(axis = 1)
            ms_rows = maxStore[state_control_columns]
            ms_sum = sum(ms_rows)
            QS_control[:,i] = sum_row/ms_sum

            ns_sub_control = numStore[state_control_columns]
            numStoreStateControl[:,i] = ns_sub_control.sum(axis = 1)

            ns_sub_treat = numStore[state_treat_columns]
            numStoreStateTreat[:,i] = ns_sub_treat.sum(axis = 1)
            
        QS_control = pd.DataFrame(QS_control)
        QS_treat = pd.DataFrame(QS_treat)
    

        numStoreStateControl = pd.DataFrame(numStoreStateControl)
        numStoreStateTreat = pd.DataFrame(numStoreStateTreat)

        salesTreatState = QS_treat
        rows_to_delete = list(range(T0,length))
        salesTreatState = salesTreatState.drop(rows_to_delete)

        salesTreatStateOOS = QS_treat
        rows_to_delete = list(range(0,T0))
        salesTreatStateOOS = salesTreatStateOOS.drop(rows_to_delete)

        salesControlState = QS_control
        rows_to_delete = list(range(T0,length))
        salesControlState = salesControlState.drop(rows_to_delete)

        salesControlStateOOS = QS_control
        rows_to_delete = list(range(0,T0))
        salesControlStateOOS = salesControlStateOOS.drop(rows_to_delete) #OOS refers to after intervention period

        #returning values
        return (n,T,numStore,sales,date,stock,state,region,TreatGroup,T0,ExogenousCovariates,salesTreat,salesControl,numTreat,numControl,ExogenousCovariatesOOS,salesTreatOOS,salesControlOOS,stateSet,numState,
                numStoreStateTreat,numStoreStateControl,salesTreatState,salesControlState,salesTreatStateOOS,salesControlStateOOS)
    

