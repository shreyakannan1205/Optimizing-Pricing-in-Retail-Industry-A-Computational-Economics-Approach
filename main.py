

from makedata import dataProduct
from farmtreat import results
import pandas as pd


def main():
    #files containing raw data
    file_1 = "main_data_prod1.csv"
    file_2 = "main_data_prod2.csv"
    file_3 = "main_data_prod3.csv"
    file_4 = "main_data_prod4.csv"
    file_5 = "main_data_prod5.csv"

                                    #price,cost,deltaPrice,tax
    #organizing data
    dataProduct_1 = dataProduct(file_1,1.89,0.7229,0.2,0.2738)
    dataProduct_2 = dataProduct(file_2,3.99,1.4891,0.2,0.2741)
    dataProduct_3 = dataProduct(file_3,1.49,0.5355,0.3,0.2708)
    dataProduct_4 = dataProduct(file_4,10.99,5.8898,-2,0.1153)
    dataProduct_5 = dataProduct(file_5,1.99,1.0475,-0.2,0.1302)

    #we are estimating optimal prices for linear trend at the municipal level
    result_1 = results(dataProduct_1,1,0)
    result_2 = results(dataProduct_2,1,0)
    result_3 = results(dataProduct_3,1,0)
    result_4 = results(dataProduct_4,1,0)
    result_5 = results(dataProduct_5,1,0)

    #optimal prices for every municipality. farm_treat.price is an array
    a1 = result_1.farmtreat_price
    a2 = result_2.farmtreat_price
    a3 = result_3.farmtreat_price
    a4 = result_4.farmtreat_price
    a5 = result_5.farmtreat_price

    #calculate average optimal price
    avg_price_1 = (sum(a1)/len(a1))[0]
    avg_price_2 = (sum(a2)/len(a2))[0]
    avg_price_3 = (sum(a3)/len(a3))[0]
    avg_price_4 = (sum(a4)/len(a4))[0]
    avg_price_5 = (sum(a5)/len(a5))[0]

    #difference between optimal price and price
    diff_1 = avg_price_1 - 1.89
    diff_2 = avg_price_2 - 3.99
    diff_3 = avg_price_3 - 1.49
    diff_4 = avg_price_4 - 10.99
    diff_5 = avg_price_5 - 1.99

    #displaying results
    result_data = {
        "Product Number" : [1,2,3,4,5],
        "Initial Price (IP)": [1.89,3.99,1.49,10.99,1.99],
        "Optimal Price (OP)": [avg_price_1,avg_price_2,avg_price_3,avg_price_4,avg_price_5],
        "Difference (OP - IP)": [diff_1,diff_2,diff_3,diff_4,diff_5]
    }

    df = pd.DataFrame(result_data)
    print(df.values)




    











    
  
    



    

    

    
main()
