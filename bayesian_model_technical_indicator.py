import numpy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import datetime

# Validation
Val = False # Need to set it to true when tuning hyperparameters

#Hyper parameter for grid search
lr = [0.01] #[0.01, 0.001]
h1 = [30] #[10, 15, 20, 25, 30]
lookback = [10] #[1, 2, 5, 7, 10]


#Multi step prediction
output_sz = 1

#Array to write the results to csv file
csv_writter = []
validation_error =[]
test_error =[]

#For early stopping
patience_threshold = 2

#Flag for index; set it to 1 when using index data
index = 1

for learning_rate in lr:
    for hidden_sz in h1:
        for input_sz in lookback:
            for run in range(30):
                print(f'###Start run {run} learning_rate {learning_rate} hidden_sz {hidden_sz}, input_sz {input_sz}')
                num_epoch = 5000 # Max value of epoch which model can be trained

                input_feature = 5
                # Step 1: Data preprocessing
                data = pd.read_csv("data/ta_data/sp500.csv")
                data_x = data.iloc[:, :-1]
                data_y = data.iloc[:, -1]

                if Val == False:
                    test_start = 2003 # Subtracting 504 days from entire data to account for 2 years
                else:
                    data_x = data_x[:2003]
                    data_y = data_y[:2003]
                    test_start = 1751

                X_train = data_x[:test_start]
                X_test = data_x[test_start:]
                Y_train = data_y[:test_start]
                Y_test = data_y[test_start:]

                # 2 scalers, one for Close and one for rest of the features
                other_column_scaler = MinMaxScaler()  # for other columns
                price_scaler = MinMaxScaler()  # for close price
                X_train[['Close']] = price_scaler.fit_transform(X_train[['Close']])
                X_test[['Close']] = price_scaler.transform(X_test[['Close']])

                if index == 1:
                    X_train[['sma_10', 'ema', 'rsi', 'mom']] = other_column_scaler.fit_transform(X_train[['sma_10', 'ema', 'rsi', 'mom']])
                    X_test[['sma_10', 'ema', 'rsi', 'mom']] = other_column_scaler.transform(X_test[['sma_10', 'ema', 'rsi', 'mom']])
                else:
                    X_train[['Volume', 'sma_10', 'ema', 'rsi', 'mom']] = other_column_scaler.fit_transform(X_train[['Volume', 'sma_10', 'ema', 'rsi', 'mom']])
                    X_train[['Open']] = price_scaler.transform(X_train[['Open']])
                    X_test[['Volume', 'sma_10', 'ema', 'rsi', 'mom']] = other_column_scaler.transform(X_test[['Volume', 'sma_10', 'ema', 'rsi', 'mom']])

                Y_train = price_scaler.transform(Y_train.values.reshape(-1, 1))
                Y_test = price_scaler.transform(Y_test.values.reshape(-1, 1))

                # building input dataset
                def x_y_split(input, label):
                    x = []
                    y = []
                    input = np.array(input)
                    label = np.array(label)
                    for i in range(input_sz, len(input) - output_sz + 1, output_sz):
                        x.append(input[i - input_sz:i].flatten())
                        y.append(label[i-1: i-1 + output_sz, 0])
                    x_np = np.array(x)
                    y_np = np.array(y)
                    x = torch.from_numpy(x_np.astype(np.float32))
                    y = torch.from_numpy(y_np.astype(np.float32))
                    return x, y


                x_train, y_train = x_y_split(X_train,Y_train)
                x_test, y_test = x_y_split(X_test,Y_test)

                y_actual = price_scaler.inverse_transform(torch.flatten(y_test).reshape(-1, 1))



                # Step 2: Building Model class
                @variational_estimator
                class BNNetwork(nn.Module):
                    def __init__(self, input_size, hidden_size):
                        super(BNNetwork, self).__init__()
                        self.bl1 = BayesianLinear(input_size, hidden_size, bias=True)
                        self.relu = nn.ReLU()
                        self.bl2 = BayesianLinear(hidden_size, output_sz)

                    def forward(self, x):
                        out = self.bl1(x)
                        out = self.relu(out)
                        out = self.bl2(out)
                        return out

                model = BNNetwork(input_feature*input_sz, hidden_sz)
                # step 3: loss & optimizer
                criterion = nn.MSELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                #print(f'model train start {datetime.datetime.now()}')
                pat = 0
                #training loop
                for epoch in range(num_epoch):
                    loss = model.sample_elbo(inputs=x_train,
                                             labels=y_train,
                                             criterion=criterion,
                                             sample_nbr=3,
                                             complexity_cost_weight=1 / x_train.shape[0])

                    # backward
                    loss.backward()

                    # updates
                    optimizer.step()
                    # zero gradient
                    optimizer.zero_grad()
                    if (epoch + 1) % 100 == 0:
                        train_MSE_err = loss.item()
                        print(f'epoch: {epoch + 1}, loss= {train_MSE_err:.4f}')

                        with torch.no_grad():
                            #print('testing the results')
                            preds = [model(x_test) for i in range(100)] #sample 100 times
                            preds = torch.stack(preds)
                            means = preds.mean(axis=0)
                            stds = preds.std(axis=0)

                            # uncertainty information
                            ci_upper_3std = means + (3 * stds)
                            ci_lower_3std = means - (3 * stds)
                            y_hat =price_scaler.inverse_transform(torch.flatten(means).reshape(-1, 1))
                            upper = price_scaler.inverse_transform(torch.flatten(ci_upper_3std).reshape(-1, 1))
                            lower = price_scaler.inverse_transform(torch.flatten(ci_lower_3std).reshape(-1, 1))

                            #Error
                            test_MSE_err = ((y_actual-y_hat)**2).mean()
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

                    # Correlation between absolute error and confidence interval
                    abs_error = abs(y_actual - y_hat).flatten()
                    ci = (upper - lower).flatten()
                    corr = np.corrcoef(abs_error, ci)
                    c = corr[0, 1]

                    #PCT
                    y_actual_next_day = y_actual[1:]
                    y_actual_today = y_actual[:-1]
                    y_hat_next_day = y_hat[1:]
                    y_hat_today = y_hat[:-1]
                    trend = ((y_actual_today-y_actual_next_day) * (y_hat_today-y_hat_next_day)) > 0
                    trend = trend.astype(int)

                    pct = np.sum(trend)/len(trend)
                    mape = np.mean(abs(y_actual-y_hat)/y_actual)*100
                    print(f' #### mape {mape}, pct {pct}  MSE ={test_MSE_err} correlation = {c}')

                # Write to CSV file
                if Val == True:
                    temp = [run, epoch, learning_rate, hidden_sz, input_sz, train_MSE_err, test_MSE_err]
                else:
                    temp = [run, train_MSE_err, test_MSE_err, mape, pct, c]
                csv_writter.append(temp)
                print(f'model train stop {datetime.datetime.now()}')
                print(f'###END run {run} epoch {epoch} learning_rate {learning_rate} hidden_sz {hidden_sz}, input_sz {input_sz}')

csv_inp = numpy.array(csv_writter)
np.savetxt("Results/test/bnn_ti_sp500.csv", csv_inp ,delimiter=",")

# #Plot test
# plt.subplot(2, 2, 1)
# x = np.arange(1, len(y_actual) + 1)
# plt.plot(x, y_actual, color='blue', label=f"Actual stock Price")
# plt.plot(x, y_hat, color='red', label=f"Predicted stock Price")
# plt.plot(x, upper, color='lightpink')
# plt.plot(x, lower, color='lightpink')
# # plt.fill_between(ci_lower_3std, ci_upper_3std, color="pink")
# plt.xlabel('Days(in future)')
# plt.ylabel("Share price")
# plt.title("Predicted & Actual Share Price")
# plt.legend()
# plt.show()