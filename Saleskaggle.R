rm(list=ls())

library(tidyverse)
library(prophet)
library(vroom)
library(lubridate)
library(tensorflow)
library(keras)

# Read in data ------------------------------------------------------------


dat <- vroom('C:/Users/daniel.feeney/OneDrive - Boa Technology Inc/Documents/Kaggle/Sales/sales_train.csv')
#head(dat)

dat$date <- dmy(dat$date)

redDat <- dat %>%
  group_by(date)%>%
  summarize(totalSale = sum(item_cnt_day))

ggplot(data = redDat, aes(x = date, y = totalSale)) + geom_point() +
  geom_line()


# Prophet -----------------------------------------------------------------
# Create data format for Prophet
pDat <- data.frame(redDat$date, redDat$totalSale)
names(pDat) <- c('ds', 'y')
  
m <- prophet(pDat)
future <- make_future_dataframe(m, periods = 100)
forecast <- predict(m, future)
plot(m, forecast)

prophet_plot_components(m, forecast)

# LSTM model --------------------------------------------------------------

# Start with overly simple model on dates rather than each store/location/item
datalags = 20
train <- pDat[1:800,]
test <- pDat[801:1000,]
batch.size = 60

x.train = array(data = lag(cbind(train$y, train$ds), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$y[-(1:datalags)], dim = c(nrow(train)-datalags, 1))
x.test = array(data = lag(cbind(test$y, test$ds), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test = array(data = test$y[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

# model in Keras
model <- keras_model_sequential()
model %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

model %>%
  compile(loss = 'mae', optimizer = 'adam')
model

for(i in 1:200){
  model %>% fit(x = x.train,
                y = y.train,
                batch_size = batch.size,
                epochs = 1,
                verbose = 0,
                shuffle = FALSE)
  model %>% reset_states()
}

pred_out <- model %>% predict(x.test, batch_size = batch.size) %>% .[,1]
