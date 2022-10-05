import numpy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime

# Validation
Val = False # Need to set it to true when tuning hyperparameters

#Hyper parameter for grid search
lr = [0.01] #[0.1, 0.01, 0.001]
h1 = [200] #[50, 100, 150, 200]
lookback = [30]


#Multi step prediction
output_sz = 1

#Array to write the results to csv file
csv_writter = []
validation_error =[]
test_error = []

#For early stopping
patience_threshold = 2

for learning_rate in lr:
    for hidden_sz in h1:
        for input_sz in lookback:
            for run in range(30): # For robustness
                print(f'###Start run {run} learning_rate {learning_rate} hidden_sz {hidden_sz}, input_sz {input_sz}')
                num_epoch = 5000 # Maximum value of epoch, update this for test data set after tuning for hyperparameters

                # Step 1: Data preprocessing

                #Function to split data into inputs and labels
                def x_y_split(ds):
                    x = []
                    y = []
                    for i in range(input_sz, len(ds) - output_sz + 1, output_sz):
                        x.append(ds[i - input_sz:i, 0])
                        y.append(ds[i: i + output_sz, 0])
                    x_np = np.array(x)
                    y_np = np.array(y)
                    x = torch.from_numpy(x_np.astype(np.float32))
                    y = torch.from_numpy(y_np.astype(np.float32))
                    return x, y

                scaler = MinMaxScaler(feature_range=(0, 1))

                #Prepare training data
                train_data = pd.read_csv("data/wrds_SP500_train.csv")
                scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))
                x_train, y_train = x_y_split(scaled_data)

                # Prepare validation/test data
                if Val == False:
                    test_data = pd.read_csv("data/wrds_SP500_test.csv")
                    scaled_test_data = scaler.transform(test_data['Close'].values.reshape(-1, 1))
                    x_test, y_test = x_y_split(scaled_test_data)
                else:
                    total_length = x_train.shape[0]
                    #Last year for validation; here x_test is x_val and y_test is y_val
                    x_test = x_train[total_length - 252:]
                    x_train = x_train[:total_length - 252]
                    y_test = y_train[total_length - 252:]
                    y_train = y_train[:total_length - 252]

                y_actual = scaler.inverse_transform(torch.flatten(y_test).reshape(-1, 1))


                # Step 2: Building Model class
                class NNetwork(nn.Module):
                    def __init__(self, input_size, hidden_size):
                        super(NNetwork, self).__init__()
                        self.l1 = nn.Linear(input_size, hidden_size, bias=True)
                        self.relu = nn.ReLU()
                        self.l2 = nn.Linear(hidden_size, output_sz)

                    def forward(self, x):
                        out = self.l1(x)
                        out = self.relu(out)
                        out = self.l2(out)
                        return out

                model = NNetwork(input_sz, hidden_sz)

                # step 3: loss & optimizer
                criterion = nn.MSELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                training_loss = []
                MSE_err_test_data= []

                # step 4: training
                print(f'model train start {datetime.datetime.now()}')
                pat = 0
                for epoch in range(num_epoch):
                    # forward pass and loss
                    y_pred = model(x_train)
                    loss = criterion(y_pred, y_train)

                    # backward
                    loss.backward()

                    # updates
                    optimizer.step()
                    # zero gradient
                    optimizer.zero_grad()

                    if (epoch + 1) % 100 == 0:
                        train_MSE_err = loss.item()
                        print(f'epoch: {epoch + 1},  loss on train data= {loss.item():.4f}')

                        with torch.no_grad():
                            y_pred = model(x_test)
                            y_hat = scaler.inverse_transform(torch.flatten(y_pred).reshape(-1, 1))

                            # Error
                            test_MSE_err = ((y_actual - y_hat) ** 2).mean()
                            print(f'test error is {test_MSE_err}')

                            if Val == True:
                                validation_error.append(test_MSE_err)
                                if len(validation_error) >= 2:
                                    if test_MSE_err > validation_error[-2]:
                                        pat = pat + 1
                                        if pat >= patience_threshold:
                                            print(f'Early stopping and test error is {test_MSE_err}')
                                            break
                            else:
                                test_error.append(test_MSE_err)

                            # only calculate for other metrics during test
                if Val == False:
                    # PCT
                    y_actual_next_day = y_actual[1:]
                    y_actual_today = y_actual[:-1]
                    y_hat_next_day = y_hat[1:]
                    y_hat_today = y_hat[:-1]
                    trend = ((y_actual_today - y_actual_next_day) * (y_hat_today - y_hat_next_day)) > 0
                    trend = trend.astype(int)

                    pct = np.sum(trend) / len(trend)
                    mape = np.mean(abs(y_actual - y_hat) / y_actual) * 100
                    print(f' #### mape {mape}, pct {pct}  MSE ={test_MSE_err}')

                # Write to CSV file
                #temp = [run, epoch, learning_rate, hidden_sz, input_sz, train_MSE_err, test_MSE_err]
                temp = [run, train_MSE_err, test_MSE_err, mape, pct]
                csv_writter.append(temp)
                # print(f'model train stop {datetime.datetime.now()}')
                print( f'###END run {run} epoch {epoch} learning_rate {learning_rate} hidden_sz {hidden_sz}, input_sz {input_sz}')

csv_inp = numpy.array(csv_writter)
np.savetxt("Results/test/fnn_close_sp500.csv", csv_inp, delimiter=",")

# #Plot test
# plt.subplot(2, 2, 1)
# x = np.arange(1, len(y_actual)+1)
# plt.plot(x, y_actual, color='blue', label=f"Actual stock Price")
# plt.plot(x,y_hat, color='red', label=f"Predicted  Price")
# plt.xlabel('Days')
# plt.ylabel("share price")
# plt.title("Predicted & Actual Share Price")
# plt.legend()
#
# plt.subplot(2, 2, 2)
# plt.plot(training_loss, color='blue', label=f"training loss")
# plt.legend()
#
# plt.subplot(2, 2, 4)
# plt.plot(MSE_err_test_data, color='blue', label=f"testing loss")
# plt.legend()
# plt.show()


