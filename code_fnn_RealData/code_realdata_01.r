library(BatchGetSymbols)
tickers <- c('FB', 'MMM')
first.date <- Sys.Date()-30
last.date <-Sys.Date()
data = BatchGetSymbols(tickers=tickers,
                       first.date = first.date,
                       last.date = last.date,do.cache=FALSE)

data$df.tickers