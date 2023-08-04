import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import Lasso
from numpy.linalg import matrix_rank
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from numpy.linalg import eig, inv
from scipy.stats import multivariate_normal
from scipy.stats import mstats
from scipy.stats import t
from makedata import dataProduct
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
import numpy.matlib

def computeBetaLasso(Y,X): #function for LASSO regression
    alpha = 1
    lasso = LassoCV(alphas=[alpha])
    lasso.fit(X, Y)
    beta = lasso.coef_
    b = np.concatenate(([lasso.intercept_], beta))
    yhat = lasso.predict(X)
    e = Y - yhat #residuals
    return [b,e]

def estimate(y,x,z,wx,wy,T0,T,flag):
    if flag == 0:
        T_star = T0
    else:
        T_star = T
    #arco estimation
    b_arco,v_arco = computeBetaLasso(y[0:T_star+1],x[0:T_star+1, :])
    b_arco = np.reshape(b_arco,(b_arco.shape[0],1))
    v_arco = np.reshape(v_arco,(v_arco.shape[0],1))

    y_arco = np.concatenate((np.ones((T, 1)), x), axis=1).dot(b_arco) # estimating counterfactual
    
    a15 = np.reshape(y[T0+1:],(y[T0+1:].shape[0],1))
    a16 = np.reshape(y_arco[T0+1:],(y_arco[T0+1:].shape[0],1))
    d_arco = a15 - a16

    # intervention effect out-of-sample (post intervention) 
    D_arco = np.mean(d_arco)  # average intervention effect out-of-sample (post intervention)
    D_arco = np.reshape(D_arco,(1,1))
    R2_arco = 1 - np.var(v_arco) / np.var(y[:T0]) 
    R2_arco = np.reshape(R2_arco,(1,1))
  

    Z = np.concatenate((np.ones((T, 1)), z),axis = 1)  # inclusion of an intercept
    n = x.shape[1]  # number of control units
    Rx = np.empty((T, n))  # memory allocation
    

    # Control units
    for i in range(n):
        if wx.size > 0:
            a = np.reshape(wx[:,i], (T,1))
            W = np.concatenate((Z, a), axis=1)
        else:
            W = Z

        WW = np.dot(W.T, W)

        if np.linalg.matrix_rank(WW) < W.shape[1]:
            W = Z
            WW = np.dot(W.T,W)

        Ax = np.linalg.solve(WW, np.dot(W.T, x[:, i]))
        Px = np.dot(W,Ax)
        Rx[:, i] = x[:, i] - Px
    Ax = np.reshape(Ax,(Ax.shape[0],1))
  
    if wy.size > 0:
        a = np.reshape(wy[:T_star+1],(wy[:T_star+1].shape[0],1))
        Z = np.concatenate((np.ones((T_star+1, 1)), z[:T_star+1, :],a), axis=1)
    else:
        Z = np.concatenate((np.ones((T_star+1, 1)), z[:T_star+1, :]), axis=1)

    ZZ = np.dot(Z.T, Z)
  

    if np.linalg.matrix_rank(ZZ) < Z.shape[1]:
        Z = np.concatenate((np.ones((T_star+1, 1)), z[:T_star+1, :]), axis=1)
        ZZ = np.dot(Z.T, Z)
        flag_rank = 1
    else:
        flag_rank = 0
    

    Ay = np.linalg.solve(ZZ, np.dot(Z.T, y[:T_star+1]))
    Ry = y[:T_star+1] - np.dot(Z, Ay)
    
   
    Ay = np.reshape(Ay,(Ay.shape[0],1))
    Ry = np.reshape(Ry,(Ry.shape[0],1))
    
    pca = PCA()  # Create a PCA object
    F = pca.fit_transform(Rx)  # Perform PCA and obtain the transformed data (scores)
    B = pca.components_ 
    B = np.delete(B,B.shape[0]-1,axis =0)

    a1 = np.reshape(Ry,(Ry.shape[0],1))
    a2 = np.empty((T-T_star-1,1))
    a3 = np.concatenate((a1,a2),axis =0)
       
    R = np.concatenate((a3, Rx), axis=1)
    n = R.shape[1]
    nz = Z.shape[1]

   
    # Select the number of components
    Omega = np.dot(Rx, Rx.T) / T_star
    Chi = np.sort(np.linalg.eigvals(Omega))[::-1]
    aux = Chi[:min(T - nz, n - nz) - 1] / Chi[1:min(T - nz, n - nz)]
    r = np.where(aux == np.max(aux))[0][0]
    r = r + 1
    Chi = np.reshape(Chi,(T,1))
    aux = np.reshape(aux,(aux.shape[0],1))
    

    # Loadings for the first unit
    FF = np.dot(F[:T_star+1, :r].T, F[:T_star+1, :r])
    FY = np.dot(F[:T_star+1, :r].T, Ry)
    b = np.linalg.solve(FF, FY)

    B = np.concatenate((b, B[:r, :]), axis=1)
   
    # Estimate counterfactual
    if flag_rank == 0:
        a5 = np.reshape(wy,(T,1))
        Z = np.concatenate((np.ones((T, 1)), z, a5), axis=1)
    else:
        Z = np.concatenate((np.ones((T, 1)), z), axis=1)


    a7 = np.dot(Z,Ay)
    a8 = np.dot(F[:, :r], B[:,0])
    a8 = np.reshape(a8,(a8.shape[0],1))
    y_pcr = a7 + a8

    d_pcr = y[T0+1:] - y_pcr[T0+1:][:,0]
    d_pcr = np.reshape(d_pcr,(d_pcr.shape[0],1))
 
    D_pcr = np.mean(d_pcr)
    D_pcr = np.reshape(D_pcr,(1,1))
    
    a9 = F[0:T0+1,0:r]
    a10 = np.dot(a9,B[0:r,0])
    a10 = np.reshape(a10,(a10.shape[0],1))
    v_pcr = Ry - a10

    R2_pcr = 1 - np.var(v_pcr) / np.var(y[:T0+1])
    R2_pcr = np.reshape(R2_pcr,(1,1))

    #FarmTreat
    a11 = np.dot(F[:,:r],B)
    U = R - a11
   
    

    #Testing for idiosyncratic contribution
    matrix_a = U[0:T_star+1,0]
    matrix_a = np.reshape(np.array(matrix_a),(matrix_a.shape[0],1))
    matrix_b = U[0:T_star+1,1:]
    matrix_c = np.matlib.repmat(matrix_a,1,n-1)
    D = matrix_c * matrix_b  

    Q = (1 / np.sqrt(T_star)) * np.sum(D, axis=0)
    Q= np.reshape(Q,(1,Q.shape[0]))

    S = np.max(np.abs(Q))
    S = np.reshape(S,(1,1))

    Boot = 500
    cov_matrix = np.cov(D, rowvar=False)
    samples = multivariate_normal.rvs(mean=np.zeros(n-1), cov=cov_matrix, size=Boot)
    Sstar = np.max(np.abs(samples), axis=1)
    Sstar = np.reshape(Sstar,(Sstar.shape[0],1))

    p_farmtreat = 1 - np.mean(S >= Sstar)
    p_farmtreat = np.reshape(p_farmtreat,(1,1))


    #Third Step: LASSO regression

    b_farmtreat,v_farmtreat = computeBetaLasso(U[:T_star+1,0],U[:T_star+1,1:])
    b_farmtreat = np.reshape(b_farmtreat,(b_farmtreat.shape[0],1))
    v_farmtreat = np.reshape(v_farmtreat,(v_farmtreat.shape[0],1))


    #Fitted Model in-sample
   
    a13 = np.reshape(wy,(T,1))
    if flag_rank == 0:
        Z = np.concatenate((np.ones((T, 1)), z, a13), axis=1)
    else:
        Z = np.concatenate((np.ones((T, 1)), z), axis=1)

 
    a14 = np.dot(Z,Ay)
    a15 = np.dot(F[:, :r], B[:, 0])
    a16 = np.dot(np.concatenate((np.ones((T, 1)), U[:, 1:]), axis=1), b_farmtreat)
    a15 = np.reshape(a15,(T,1))
    y_farmtreat = a14 + a15 + a16

    a17 = np.reshape(y[T0+1:],(y[T0+1:].shape[0],1))
    a18 = np.reshape(y_farmtreat[T0+1:],(y_farmtreat[T0+1:].shape[0],1))

    d_farmtreat = a17 - a18

    
    D_farmtreat = np.mean(d_farmtreat)
    D_farmtreat = np.reshape(D_farmtreat,(1,1))
    

    R2_farmtreat = 1 - np.var(v_farmtreat) / np.var(y[:T0])
    R2_farmtreat = np.reshape(R2_farmtreat,(1,1))

    
    d_arco = d_arco.reshape(d_arco.shape[0])
    D_arco = D_arco[0][0]
    R2_arco = R2_arco[0][0]
    v_arco = v_arco.reshape(v_arco.shape[0])
    y_arco = y_arco.reshape(y_arco.shape[0])
    d_pcr = d_pcr.reshape(d_pcr.shape[0])
    D_pcr = D_pcr[0][0]
    R2_pcr = R2_pcr[0][0]
    v_pcr = v_pcr.reshape(v_pcr.shape[0])
    y_pcr = y_pcr.reshape(y_pcr.shape[0])
    d_farmtreat = d_farmtreat.reshape(d_farmtreat.shape[0])
    D_farmtreat = D_farmtreat[0][0]
    R2_farmtreat = R2_farmtreat[0][0]
    v_farmtreat = v_farmtreat.reshape(v_farmtreat.shape[0])
    y_farmtreat = y_farmtreat.reshape(y_farmtreat.shape[0])
    p_farmtreat = p_farmtreat[0][0]

    return [d_arco,D_arco,R2_arco,v_arco,y_arco, d_pcr,D_pcr,R2_pcr,v_pcr,y_pcr, 
    d_farmtreat,D_farmtreat,R2_farmtreat,v_farmtreat,y_farmtreat,p_farmtreat,r]


def ressampling(fun,v,T0,T2,x,tau):
    if x.shape == ():
        x = np.reshape(x,(1,1))
        
    else: 
        x = np.reshape(x,(x.shape[0],1))

    phi = np.full((T0 - T2 + 3, x.shape[0]), np.nan)

    for j in range(T0 - T2 + 3):
        phi[j, :] = fun((v[j:j+T2-1])).T

    Q1 = np.sum(phi <= np.abs(x.T)) / (T0 - T2 + 3)
    Q2 = np.sum(phi <= -np.abs(x.T)) / (T0 - T2 + 3)
    p = 1 - Q1 + Q2
    lb = []
    ub = []
    if tau:
        lb = np.quantile(phi, tau/2)
        ub = np.quantile(phi, 1 - tau/2)
    return [p,ub,lb]

def farmTreat(dataProduct_1,flag_trend,flag_state):
    if flag_state == 0:
        numStore = dataProduct_1.numStore
        numTreat = dataProduct_1.numTreat
        y        = dataProduct_1.salesTreat.values
        yOOS     = dataProduct_1.salesTreatOOS.values
        x        = dataProduct_1.salesControl.values
        xOOS     = dataProduct_1.salesControlOOS.values
        z        = dataProduct_1.ExogenousCovariates.values
        col_add = list(range(0,dataProduct_1.T0))
        col_add = np.reshape(col_add,(len(col_add),1))
        z = np.concatenate((z,col_add),axis = 1)
        zOOS     = dataProduct_1.ExogenousCovariatesOOS.values
        col_add1 = list(range(dataProduct_1.T0,dataProduct_1.T))
        col_add1 = np.reshape(col_add1,(len(col_add1),1))
        zOOS = np.concatenate((zOOS,col_add1),axis = 1)
      
        T        = dataProduct_1.T
        T0       = dataProduct_1.T0
        gtreat   = dataProduct_1.TreatGroup

        columns_control = gtreat.columns[gtreat.iloc[0] == False].to_list()
        wx = numStore[columns_control].values

        columns_treat = gtreat.columns[gtreat.iloc[0] == True].to_list()
        wy = numStore[columns_treat].values
        gtreat = gtreat.values
        numStore = numStore.values

    else:
        numTreat = dataProduct_1.numState
        y        = dataProduct_1.salesTreatState.values
        yOOS     = dataProduct_1.salesTreatStateOOS.values
        x        = dataProduct_1.salesControlState.values
        xOOS     = dataProduct_1.salesControlStateOOS.values
        z        = dataProduct_1.ExogenousCovariates.values
        zOOS     = dataProduct_1.ExogenousCovariatesOOS.values
        T        = dataProduct_1.T
        T0       = dataProduct_1.T0
        
        wx = dataProduct_1.numStoreStateControl.values
        wy = dataProduct_1.numStoreStateTreat.values

    d_arco     = np.empty((T-T0,numTreat))
    D_arco     = np.empty((numTreat,1))    
    R2_arco    = np.empty((numTreat,1))     
    v_arco     = np.empty((T0,numTreat))   
    y_arco     = np.empty((T,numTreat))      
    elast_arco = np.empty((numTreat,1))  
    price_arco = np.empty((numTreat,1))   
    

    d_pcr     = np.empty((T-T0,numTreat))
    D_pcr     = np.empty((numTreat,1))
    R2_pcr    = np.empty((numTreat,1))
    v_pcr     = np.empty((T0,numTreat))
    y_pcr     = np.empty((T,numTreat))
    elast_pcr = np.empty((numTreat,1))
    price_pcr = np.empty((numTreat,1))

    d_farmtreat     = np.empty((T-T0,numTreat))
    D_farmtreat     = np.empty((numTreat,1))
    R2_farmtreat    = np.empty((numTreat,1))
    v_farmtreat     = np.empty((T0,numTreat))
    y_farmtreat     = np.empty((T,numTreat))
    elast_farmtreat = np.empty((numTreat,1))
    price_farmtreat = np.empty((numTreat,1))


    p_arco  = np.empty((T-T0,numTreat))    
    p2_arco = np.empty((numTreat,1))        
    p3_arco = np.empty((numTreat,1))       
    ub_arco = np.empty((T-T0,numTreat))  
    lb_arco = np.empty((T-T0,numTreat))  

    p_pcr  = np.empty((T-T0,numTreat))
    p2_pcr = np.empty((numTreat,1))
    p3_pcr = np.empty((numTreat,1))
    ub_pcr = np.empty((T-T0,numTreat))
    lb_pcr = np.empty((T-T0,numTreat))


    p_farmtreat  = np.empty((T-T0,numTreat))
    p2_farmtreat = np.empty((numTreat,1))
    p3_farmtreat = np.empty((numTreat,1))
    p4_farmtreat = np.empty((numTreat,1))
    ub_farmtreat = np.empty((T-T0,numTreat))
    lb_farmtreat = np.empty((T-T0,numTreat))

    numFactors = np.empty((numTreat,1))

    if flag_trend == 0:
        z = z[:, 1:]
        zOOS = zOOS[:, 1:]

    for i in range(numTreat):
        print(i)
        if np.isnan(np.sum(y[:, i])) == False:
            z_new = np.concatenate((z,zOOS))
            d_arco[:,i], D_arco[i][0], R2_arco[i][0], v_arco[:,i], y_arco[:,i], d_pcr[:,i], D_pcr[i][0],R2_pcr[i][0], v_pcr[:,i],y_pcr[:,i],d_farmtreat[:,i],D_farmtreat[i][0],R2_farmtreat[i][0],v_farmtreat[:,i],y_farmtreat[:,i],p4_farmtreat[i][0],numFactors[i][0] = estimate(np.concatenate((y[:, i], yOOS[:, i])), np.concatenate((x, xOOS)), z_new, wx, wy[:, i], T0-1, T, 0)
           
            p_arco[:, i], ub_arco[:, i], lb_arco[:, i] = ressampling(lambda x: x, v_arco[:, i], T0-1, T-T0+1, d_arco[:, i], 0.05)
            p2_arco[i, 0] = ressampling(lambda x: np.mean(x**2), v_arco[:, i], T0-1, T-T0+1, np.mean(d_arco[:, i]**2), 0.05)[0]
            p3_arco[i, 0] = ressampling(lambda x: np.mean(np.abs(x)), v_arco[:, i], T0-1, T-T0+1, np.mean(np.abs(d_arco[:, i])), 0.05)[0]
        
            # Resampling for v_pcr
            p_pcr[:, i], ub_pcr[:, i], lb_pcr[:, i] = ressampling(lambda x: x, v_pcr[:, i], T0-1, T-T0+1, d_pcr[:, i], 0.05)
            p2_pcr[i, 0] = ressampling(lambda x: np.mean(x**2), v_pcr[:, i], T0-1, T-T0+1, np.mean(d_pcr[:, i]**2), 0.05)[0]
            p3_pcr[i, 0] = ressampling(lambda x: np.mean(np.abs(x)), v_pcr[:, i], T0-1, T-T0+1, np.mean(np.abs(d_pcr[:, i])), 0.05)[0]
            
            # # Resampling for v_farmtreat
            p_farmtreat[:, i], ub_farmtreat[:, i], lb_farmtreat[:, i] = ressampling(lambda x: x, v_farmtreat[:, i], T0-1, T-T0+1, d_farmtreat[:, i], 0.05)
            p2_farmtreat[i, 0] = ressampling(lambda x: np.mean(x**2), v_farmtreat[:, i], T0-1, T-T0+1, np.mean(d_farmtreat[:, i]**2), 0.05)[0]
            p3_farmtreat[i, 0] = ressampling(lambda x: np.mean(np.abs(x)), v_farmtreat[:, i], T0-1, T-T0+1, np.mean(np.abs(d_farmtreat[:, i])), 0.05)[0]

            b_arco = D_arco[i, 0] / dataProduct_1.deltaPrice
            b_pcr = D_pcr[i, 0] / dataProduct_1.deltaPrice
            b_farmtreat = D_farmtreat[i, 0] / dataProduct_1.deltaPrice

            elast_arco[i, 0] = (b_arco * dataProduct_1.price) / np.mean(y_arco[T0:, i])
            elast_pcr[i, 0] = (b_pcr * dataProduct_1.price) / np.mean(y_pcr[T0:, i])
            elast_farmtreat[i, 0] = (b_farmtreat * dataProduct_1.price) / np.mean(y_farmtreat[T0:, i])

            price_arco[i, 0] = ((1 - dataProduct_1.tax) * (np.mean(y_arco[T0:, i]) - b_arco * dataProduct_1.price) - b_arco * dataProduct_1.cost) / (-2 * b_arco * (1 - dataProduct_1.tax))
            price_pcr[i, 0] = ((1 - dataProduct_1.tax) * (np.mean(y_pcr[T0:, i]) - b_pcr * dataProduct_1.price) - b_pcr * dataProduct_1.cost) / (-2 * b_pcr * (1 - dataProduct_1.tax))
            price_farmtreat[i, 0] = ((1 - dataProduct_1.tax) * (np.mean(y_farmtreat[T0:, i]) - b_farmtreat * dataProduct_1.price) - b_farmtreat * dataProduct_1.cost) / (-2 * b_farmtreat * (1 - dataProduct_1.tax))
            

    return [y_arco,d_arco,D_arco,R2_arco,lb_arco,ub_arco,p_arco,p2_arco,p3_arco,elast_arco,price_arco,
            y_pcr,d_pcr,D_pcr,R2_pcr,lb_pcr,ub_pcr,p_pcr,p2_pcr,p3_pcr,elast_pcr,price_pcr,
            y_farmtreat,d_farmtreat,D_farmtreat,R2_farmtreat,lb_farmtreat,ub_farmtreat,p_farmtreat,p2_farmtreat,p3_farmtreat,p4_farmtreat,elast_farmtreat,price_farmtreat,numFactors]


class results:

    def __init__(self,dataProduct_1,flag_trend, flag_state):

        res = farmTreat(dataProduct_1,flag_trend,flag_state)
        self.arco_sales = res[0]
        self.arco_delta = res[1]
        self.arco_Delta = res[2]
        self.arco_R2 = res[3]
        self.arco_lowerIC = res[4]
        self.arco_upperIC = res[5]
        self.arco_pvalue = res[6]
        self.arco_pvalueSQR = res[7]
        self.arco_pvalueABS = res[8]
        self.arco_elasticity= res[9]
        self.arco_price = res[10]

        self.pcr_sales = res[11]
        self.pcr_delta = res[12]
        self.pcr_Delta = res[13]
        self.pcr_R2 = res[14]
        self.pcr_lowerIC = res[15]
        self.pcr_upperIC = res[16]
        self.pcr_pvalue = res[17]
        self.pcr_pvalueSQR = res[18]
        self.pcr_pvalueABS = res[19]
        self.pcr_elasticity= res[20]
        self.pcr_price = res[21]

        self.farmtreat_sales = res[22]
        self.farmtreat_delta = res[23]
        self.farmtreat_Delta = res[24]
        self.farmtreat_R2 = res[25]
        self.farmtreat_lowerIC = res[26]
        self.farmtreat_upperIC = res[27]
        self.farmtreat_pvalue = res[28]
        self.farmtreat_pvalueSQR = res[29]
        self.farmtreat_pvalueABS = res[30]
        self.farmtreat_pvalueCOV = res[31]
        self.farmtreat_elasticity= res[32]
        self.farmtreat_price = res[33]

        self.numFactors = res[34]
        