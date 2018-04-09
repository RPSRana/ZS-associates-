# This starter script is meant to help you understand how you can make your first submission 
# in the format as expected. This scripts predicts events based on the popular past events per patient.

path <- "directory"
setwd("E:/ZS competition")
install.packages("data.table")
# load library
library(data.table)


# load and check data ---------------------------------------------------------------

train <- fread("train.csv")
test <- fread("test.csv")
sample_sub <- fread("sample_submission.csv")

head(train)
head(test)

str(train)
str(test)

# order data
train <- train[order(PID)]
test<- test[order(PID)]



# Predicting future events based on popular past events per patient -------
train_dcast <- dcast(data = train, PID ~ Event, length, value.var = "Event")

# get top 10 events per row
random_submit <- colnames(train_dcast)[-1][apply(train_dcast[,-c('PID'),with=F],1, function(x)order(-x)[1:10])]

# create the submission file
random_mat <- as.data.table((matrix(random_submit,ncol = 10, byrow = T)))
colnames(random_mat) <- colnames(sample_sub)[-1]
random_mat <- cbind(PID = test$PID, random_mat)
fwrite(random_mat,"random_sub.csv")


library(markovchain)
list_train <- train[,.(list(Event)),.(PID,Date)]
list_one <- list_train[,.(list(V1)),.(PID)]
list_one[,V1 := lapply(V1, unlist, use.names = F)]
setnames(list_one,"V1","Events")
prediction <- list()
for(x in 1:nrow(list_one))
{
  PID <- list_one[x,PID]
  events_x <- as.character(unlist(list_one[x,Events]))
  
  mcX <- markovchainFit(events_x, method = "bootstrap", nboot = 50)
  mcX$estimate
  pred <- predict(object = mcX$estimate, newdata = events_x, n.ahead=10) # predict next 10 events
  
  prediction[[PID]] <- pred
  
}

