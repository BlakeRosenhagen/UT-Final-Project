
import pandas as pd

def get_data():
   
    data = pd.read_csv('data/transactions.csv')
    X = data.loc[(data.type == 'TRANSFER') | (data.type == 'CASH_OUT')]
    Y = X['isFraud']
    del X['isFraud']

    #columns that are not useful
    X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

   
    X['type'] = X['type'].replace({'TRANSFER': 0, 'CASH_OUT': 1})
###I got help from stack overflow on this part
    X.drop(X.loc[(X.oldbalanceDest == 0) & (X.newbalanceDest == 0) & (X.amount != 0)], axis=1)
    X.drop(X.loc[(X.oldbalanceOrg == 0) & (X.newbalanceOrig == 0) & (X.amount != 0)], axis=1)
###end
    return data, X, Y

if __name__ == '__main__':
    d, x, y=get_data()
    print("Data Preparation Complete")
