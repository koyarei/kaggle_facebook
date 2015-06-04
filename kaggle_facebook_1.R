# set up file and load saved data
setwd("/home/rstudio/Dropbox/Kaggle_Facebook")

# bids <- read.csv("bids.csv")
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# bids.full <- merge(bids, train)
# bids.w.test <- merge(bids, test)
# save(bids.full, file="bids.full.RData", ascii=T, compress=T)
# save(bids.w.test, file="bids.w.test.RData", ascii=T, compress=T)
load("bids.full.RData")
load("bids.w.test.RData")

install.packages("caret")
library(caret)
install.packages("pROC")
library(pROC)
install.packages("e1071")
library(e1071)
install.packages("randomForest")
library(randomForest)
install.packages("gbm")
library(gbm)
install.packages("ROCR")
library(ROCR)
install.packages("Hmisc")
library(Hmisc)
install.packages("mlbench")
library(mlbench)
install.packages("party")
library(party)
install.packages("doParallel")
library(doParallel)


# extract basically volume features
# save(bidder.train, file="bidder_train.RData", ascii=T, compress=T)
load("bidder_train.RData")
# save(bidder.train.ext, file="bidder_train_ext.RData", ascii=T, compress=T)
load("bidder_train_ext.RData")
# save(bidder.test, file="bidder.test.RData", ascii=T, compress=T)
load("bidder.test.RData")
# save(bidder.test.ext, file="bidder_test_ext.RData", ascii=T, compress=T)
load("bidder_test_ext.RData")

# extracted per auction feature
# save(bidder.train.ext.3, file = 'bidder_train_ext_3.RData', ascii = T, compress = T)
# save(bidder.test.ext.3, file = 'bidder_test_ext_3.RData', ascii = T, compress = T)
load('bidder_train_ext_3.RData')
load('bidder_test_ext_3.RData')

# extracted time differences for per auction (16 features)
# save(bidder.train.ext.4, file = "bidder.train.ext.4.RData", ascii = T, compress = T)
# save(bidder.test.ext.4, file = "bidder.test.ext.4.RData", ascii = T, compress = T)
load("bidder.train.ext.4.RData")
load("bidder.test.ext.4.RData")

# save(bidder.train.ext.5, file = 'bidder.train.ext.5.RData')
# save(bidder.test.ext.5, file = 'bidder.test.ext.5.RData')
load('bidder.train.ext.5.RData')
load('bidder.test.ext.5.RData')

# save file so running rf and gbm model to get varImp is not needed every time
# save(merged.import, file = "merged.import.RData")
load("merged.import.RData")
# most important features according to caret RF only
load('rf.import.RData')

# save(bidder.train.ext.6, file = 'bidder.train.ext.6.Rdata')
# save(bidder.test.ext.6, file = 'bidder.test.ext.6.RData')
load('bidder.train.ext.6.Rdata')
load('bidder.test.ext.6.RData')

# extract bidder information; transform training data to be bidder per row
# features to be extracted: 
#   number of auctions per bidder
#   bids:
#       number of bids per bidder
#       min, max, mean, and sd of bids per auction per bidder
#   merchandise:
#       number of merchandises per bidder
#       min, max, mean, and sd of merchandises per auction per bidder
#   device:
#       number of devices per bidder
#       min, max, mean, and sd of devices per auction per bidder
#   country:
#       number of countries per bidder
#       min, max, mean, and sd of countries per auction per bidder
#   ip:
#       number of IPs per bidder
#       min, max, mean, and sd of IPs per auction per bidder
#   time:
#       smallest time difference per auction for any auction per bidder
#       largest time difference per auction for any auction per bidder
#       standard deviation of time differences per auction per bidder
# #       mean of time differences per auction per bidder
# 
bidder.train <- data.frame() 
# this function extracts information of number of bids, merchandise, devices, countries,
# ips, and store in a data.frame with every bidder_id in an individual row
GetFactors <- function(bidder.id) {
  # the function creates a subset based on the bidder_id, and then extracts number of
  # merchandise, device, country, ip, payment_account, address, url, and outcome from the dataset
  # Args:
  #   bidder_id: the identifier used to create subsets
  subset <- subset(bids.full, as.character(bidder_id) == as.character(bidder.id))
  bids <- nrow(subset)
  c <- c("auction","merchandise", "device", "country", "ip", "url", "payment_account", "address", "outcome")
  
  output <- data.frame(bidder_id=subset[1,1])
  
  list <- c()
  for (i in 1:(length(c) - 1)) {
    list <- c(list,nrow(unique(subset[c[i]])))
  }
  
  list <- c(list, mean(summary(subset$outcome)))
  
  # add number of bids
  list <- c(bids, list)

#   print(names(list))
  names(list) <- c("bids", c)
  list <- t(data.frame(list))
  output <- cbind(output, list)
  bidder.train <<- rbind(bidder.train, output)
  return(bidder.train)
    
}

tapply(bids.full$bidder_id, bids.full$bidder_id, GetFactors)
# change outcome to proper factor names
bidder.train[bidder.train$outcome == 1, "outcome"] <- "X1"
bidder.train[bidder.train$outcome == 0, "outcome"] <- "X0"
rownames(bidder.train) <- NULL
# save(bidder.train, file="bidder_train.RData", ascii=T, compress=T)
load("bidder_train.RData")

## more feature extraction
bidder.aucs <- data.frame()
# set a counter to know the progress of function process
count <- length(bidder.train$bidder_id)
# this function extracts min, max, mean, and sd for each of the auction associated with each bidder,
# and stores the features in a data.frame with each bidder_id in a single row
GetPerAuctionFeatures <- function(bidder.id) {
# the function is intended to use combined with tapply; it iterates through the entire data.frame,
# gets auctions associated with each bidder_id, and creates a data.frame with each row representing
# an auction
# the function updates the global variable (data.frame) bidder.aucs
  bidder.subset <- subset(bids.full, bidder_id == bidder.id)
  bidder.subset.aucs <- data.frame()
  tapply(bidder.subset$auction, bidder.subset$auction, function(x) {
    # anonymous function to extract min, max, mean of auctions, merchandise,
    # device, country, ip, and url per bid
    auc.subset <- subset(bidder.subset, auction == x)
    
    
    c <- c("bid_id", "device", "country", "ip", "url")
    # iterate over the list c to get min, max, mean, and sd for each of the features per auction
    l <- c()
    for (i in 1:length(c)) {
      num.row <- nrow(unique(auc.subset[c[i]]))
      l <- c(l, num.row)
    }
    
    # give l a bidder_id feature
    l[6] <- as.character(subset(bids.full, as.character(auction) == as.character(x))$bidder_id)
    l <- data.frame(l)
    l.row <- data.frame(t(l))
    print(class(l.row))
    print(l.row)
    bidder.subset.aucs <<- rbind(bidder.subset.aucs, l.row)
   
#     bidder.aucs <<- rbind(bidder.aucs, l.row)
    
  })
  names(bidder.subset.aucs) <- c("bids.per.auc", "devices.per.auc",
                               "countries.per.auc", "ips.per.auc", "urls.per.auc", 
                               "bidder_id")

  GetMinMaxMeanSd <- function(df) {
  # a function to extract min, max, mean, and sd from each column of a data.frame
  #  Args:
  #   df: a data.frame
  #   return a one-row data.frame with min, max, mean, and sd for each original column feature
    auc.df <- data.frame(bidder_id = bidder.id)
    for (i in 1:(ncol(df)-1)) {
      
      min <- min(as.numeric(df[,i]))
      max <- max(as.numeric(df[,i]))
      mean <- mean(as.numeric(df[,i]))
      sd <- sd(as.numeric(df[,i]))
      row <- c(min, max, mean, sd)
      row <- data.frame(t(data.frame(row)))
      feature <- strsplit(names(df)[i], "\\.")[[1]][1]
      name.min <- paste0(feature, ".", "min")
      name.max <- paste0(feature, ".", "max")
      name.mean <- paste0(feature, ".", "mean")
      name.sd <- paste0(feature, ".", "sd")
      names(row) <- c(name.min, name.max, name.mean, name.sd)
      auc.df <- cbind(auc.df, row)
      
    }
    return(auc.df)
  
  }
  
  auc.df <- GetMinMaxMeanSd(bidder.subset.aucs)
  bidder.aucs <<- rbind(bidder.aucs, auc.df)
  
  count <<- count - 1
  print(count)
  return(bidder.aucs)
}
tapply(bidder.train$bidder_id, as.factor(bidder.train$bidder_id), GetPerAuctionFeatures)
# bidder.aucs <- unique(bidder.aucs)
rownames(bidder.aucs) <- NULL
# save(bidder.aucs, file="bidder_train_aucs.RData", ascii=T, compress=T)
load("bidder_train_aucs.RData")

bidder.train.ext <- merge(bidder.train, bidder.aucs)
# save(bidder.train.ext, file="bidder_train_ext.RData", ascii=T, compress=T)
load("bidder_train_ext.RData")


# prepare the final testing set
bidder.test <- data.frame()
GetFactorsTest <- function(bidder.id) {
  # the function creates a subset based on the bidder_id, and then extracts number of
  # merchandise, device, country, ip, payment_account, address, url, and outcome from the dataset
  # Args:
  #   bidder_id: the identifier used to create subsets
  subset <- subset(bids.w.test, as.character(bidder_id) == as.character(bidder.id))
  bids <- nrow(subset)
  c <- c("auction","merchandise", "device", "country", "ip", "url", "payment_account", "address", "outcome")
  
  output <- data.frame(bidder_id=subset[1,1])
  
  list <- c()
  for (i in 1:(length(c) - 1)) {
    list <- c(list,nrow(unique(subset[c[i]])))
  }
  
  list <- c(list, mean(summary(subset$outcome)))
  
  # add number of bids
  list <- c(bids, list)
  
  #   print(names(list))
  names(list) <- c("bids", c)
  list <- t(data.frame(list))
  output <- cbind(output, list)
  bidder.test <<- rbind(bidder.test, output)
  return(bidder.test)
  
}
tapply(bids.w.test$bidder_id, bids.w.test$bidder_id, GetFactorsTest)
# keep bidder.test dataset
# save(bidder.test, file="bidder.test.RData", ascii=T, compress=T)
load("bidder.test.RData")

## more feature extraction - test
bidder.aucs.test <- data.frame()
# set a counter to know the progress of function process
count.test <- length(bidder.test$bidder_id)
# this function extracts min, max, mean, and sd for each of the auction associated with each bidder,
# and stores the features in a data.frame with each bidder_id in a single row
GetPerAuctionFeaturesTest <- function(bidder.id) {
  # the function is intended to use combined with tapply; it iterates through the entire data.frame,
  # gets auctions associated with each bidder_id, and creates a data.frame with each row representing
  # an auction
  # the function updates the global variable (data.frame) bidder.aucs
  bidder.subset <- subset(bids.w.test, bidder_id == bidder.id)
  bidder.subset.aucs <- data.frame()
  
  lapply(split(bidder.subset, as.factor(as.character(bidder.subset$auction))), function(auc.subset) {
    
    
    c <- c("bid_id", "device", "country", "ip", "url")
    # iterate over the list c to get min, max, mean, and sd for each of the features per auction
    l <- c()
    for (i in 1:length(c)) {
      num.row <- nrow(unique(auc.subset[c[i]]))
      l <- c(l, num.row)
    }
    
    # give l a bidder_id feature
    l[6] <- as.character(auc.subset$bidder_id[])
    l <- data.frame(l)
    l.row <- data.frame(t(l))
    print(class(l.row))
    print(l.row)
    bidder.subset.aucs <<- rbind(bidder.subset.aucs, l.row)
    
    
    
  })
  
  

  names(bidder.subset.aucs) <- c("bids.per.auc", "devices.per.auc",
                                 "countries.per.auc", "ips.per.auc", "urls.per.auc", 
                                 "bidder_id")
  
  GetMinMaxMeanSd <- function(df, bidder.id) {
    # a function to extract min, max, mean, and sd from each column of a data.frame
    #  Args:
    #   df: a data.frame
    #   return a one-row data.frame with min, max, mean, and sd for each original column feature
    auc.df <- data.frame(bidder_id = bidder.id)
    for (i in 1:(ncol(df)-1)) {
      
      min <- min(as.numeric(df[,i]))
      max <- max(as.numeric(df[,i]))
      mean <- mean(as.numeric(df[,i]))
      sd <- sd(as.numeric(df[,i]))
      row <- c(min, max, mean, sd)
      row <- data.frame(t(data.frame(row)))
      feature <- strsplit(names(df)[i], "\\.")[[1]][1]
      name.min <- paste0(feature, ".", "min")
      name.max <- paste0(feature, ".", "max")
      name.mean <- paste0(feature, ".", "mean")
      name.sd <- paste0(feature, ".", "sd")
      names(row) <- c(name.min, name.max, name.mean, name.sd)
      auc.df <- cbind(auc.df, row)
      
    }
    return(auc.df)
    
  }
  
  auc.df <- GetMinMaxMeanSd(bidder.subset.aucs, bidder.id)
  bidder.aucs.test <<- rbind(bidder.aucs.test, auc.df)
  
  count.test <<- count.test - 1
  print(count.test)
  return(bidder.aucs)
}
tapply(bidder.test$bidder_id, as.factor(bidder.test$bidder_id), GetPerAuctionFeaturesTest)
# bidder.aucs <- unique(bidder.aucs)
rownames(bidder.aucs.test) <- NULL
# save(bidder.aucs.test, file="bidder_aucs_test.RData", ascii=T, compress=T)
load("bidder_aucs_test.RData")

bidder.test.ext <- merge(bidder.test, bidder.aucs.test)
# save(bidder.test.ext, file="bidder_test_ext.RData", ascii=T, compress=T)
load("bidder_test_ext.RData")
# change sd with value NA to be 0
ModifySd <- function(df) {
  # this function takes a data.frame as input, find the columns that include name 'sd',
  # and replace all the NAs with 0
  # Args:
  #   df: a data.frame
  # returns: a data.frame with standard deviations modified to include no NAs.
  sd.c <- c("bids.sd", "devices.sd", "countries.sd", "ips.sd", "urls.sd")
  lapply(sd.c, function(col) {
    df[is.na(df[, col]), col] <<- 0
  })
  
  return(df)
  
}

# apply to training set
bidder.train.ext.2 <- ModifySd(bidder.train.ext)
# apply to testing set: replace standard deviations NAs with 0
bidder.test.ext.2 <- ModifySd(bidder.test.ext)


# extract merchandise information, from number to actual category
# there is one bidder_id with two merchandises
two.merc.bidder <- bidder.train.ext[which.max(bidder.train.ext$merchandise), 'bidder_id']
two.merc.bidder.full <- subset(bids.full, bidder_id == two.merc.bidder)
# if there are more than one merchandise categories, pick the more frequent option
top.merc <- which.max(summary(two.merc.bidder.full$merchandise))
bids.full.merc.modified <- bids.full
bids.full.merc.modified[bids.full.merc.modified$bidder_id == two.merc.bidder, 'merchandise'] <- as.factor(names(top.merc))

# modify the bidder.train.ext dataset to include merchandise names
bidder.merc <- unique(bids.full.merc.modified[, c('bidder_id', 'merchandise')])
bidder.train.ext.3 <- merge(bidder.train.ext.2[, names(bidder.train.ext) != 'merchandise'], bidder.merc)

# modify the bidder.test.ext dataset to include merchandise names
bidder.merc.test <- unique(bids.w.test[, c('bidder_id', 'merchandise')])
bidder.test.ext.3 <- merge(bidder.test.ext.2[, names(bidder.test.ext) != 'merchandise'], bidder.merc.test)

# save(bidder.train.ext.3, file = 'bidder_train_ext_3.RData', ascii = T, compress = T)
# save(bidder.test.ext.3, file = 'bidder_test_ext_3.RData', ascii = T, compress = T)
load('bidder_train_ext_3.RData')
load('bidder_test_ext_3.RData')

# extract time information
# extract time differences between bids in any given auction
# extract the min, max, mean, and sd of time differences across all auctions
bidder.time <- data.frame()
count.time <- nrow(bidder.train.ext.3)
GetTimeDiff <- function(bidder.id) {
  # the function is used in conjuction with tapply through either bids.full or bids.w.test
  # it iterates through each bidder_id, finds all associated auctions for that bidder,
  # first finds min, mean, max, and sd of all time differences between each bid in a given auction,
  # second it calculates the min, mean, max, and sd of the above values for any given auction, across
  # all auctions.
  # Eventually the function will update the bidder.time data.frame to include 16 time related features.
  #
  # Args:
  #   bidder.id: the bidder_id factor; to be used as a key to iterate through in tapply
  #
  # returns: nothing;
  # but through each iteration, bidder.time is updated;
  # bidder.time will have the following features:
  # mean.min.diff.auc, max.min.diff.auc, min.min.diff.auc, sd.min.diff.auc,
  # mean.mean.diff.auc, max.mean.diff.auc, min.mean.diff.auc, sd.mean.diff.auc,
  # mean.max.diff.auc, max.max.diff.auc, min.max.diff.auc, sd.max.diff.auc,
  # mean.sd.diff.auc, max.sd.diff.auc, min.sd.diff.auc, sd.sd.diff.auc
  bidder.subset <- subset(bids.full, bidder_id == bidder.id)
  bidder.subset <- bidder.subset[order(bidder.subset$time), ]
  bidder.subset.aucs <- data.frame()
  lapply(split(bidder.subset, as.factor(as.character(bidder.subset$auction))), function(auc.subset) {
      # the lapply iterates through each of the auction subset,
      # and extracts min, mean, max, and sd of bid time difference for each auction
      times <- sort(auc.subset$time)
      if (length(times) > 1) {
        new.times <- times[2:length(times)]
      } else {
        # if only one bid is found for an auction, get time from the first bid of next auction
        # if only one bid is found for an user, or there is no subsequent bids, 
        #  get next bid time from the max value of the dataset
        new.times <- bidder.subset[grep(auc.subset$auction[1], bidder.subset$auction) + 1, 'time'] 
        if (is.na(new.times)) {
          new.times <- max(bids.full$time)
        }
      
      }
      diff <- new.times - times[1:(length(times) - 1)]
      sd <- ifelse(is.na(sd(diff)), 0, sd(diff) )
      row <- c(min(diff), mean(diff), max(diff), sd)
      
      
      # handle only 1 bid in the auction
     
      row <- data.frame(t(data.frame(row)))
      row[5] <- as.character(auc.subset$auction[1])
      row[6] <- as.character(bidder.id)
      names(row) <- c("min.diff.auc", "mean.diff.auc", "max.diff.auc", "sd.diff.auc", "auction", "bidder_id")
      bidder.subset.aucs <<- rbind(bidder.subset.aucs, row)
      
      })
    #   # if only one bid is found, fill in the min, mean, max with mean of time difference across all auctions
    #   # if sd.diff.auc is mean for an auction, fill in 0
    #   mean.all <- mean(bidder.subset.aucs[!is.na(bidder.subset.aucs$mean.diff.auc), 'mean.diff.auc'])
    #   bidder.subset.aucs[is.na(bidder.subset.aucs[,1]), 1] <- mean.all
    #   bidder.subset.aucs[is.na(bidder.subset.aucs[,2]), 2] <- mean.all
    #   bidder.subset.aucs[is.na(bidder.subset.aucs[,3]), 3] <- mean.all
    #   bidder.subset.aucs[is.na(bidder.subset.aucs[,4]), 4] <- 0
      # now extract the second layer of information: the distribution of the summary metrics
      GetMinMaxMeanSd <- function(df) {
      # a function to extract min, max, mean, and sd from each column of a data.frame
      #  Args:
      #   df: a data.frame
      #   return a one-row data.frame with min, max, mean, and sd for each original column feature
        auc.df <- data.frame(bidder_id = bidder.id)
        for (i in 1:(ncol(df)-1)) {
          min <- min(as.numeric(df[,i]))
          max <- max(as.numeric(df[,i]))
          mean <- mean(as.numeric(df[,i]))
          sd <- sd(as.numeric(df[,i]))
          sd <- ifelse(is.na(sd), 0, sd)
          row <- c(min, max, mean, sd)
          row <- data.frame(t(data.frame(row)))
          feature <- names(df)[i]
          name.min <- paste0("min", ".", feature)
          name.mean <- paste0("mean", ".", feature)
          name.max <- paste0("max", ".", feature)
          name.sd <- paste0("sd", ".", feature)
          names(row) <- c(name.min, name.mean, name.max, name.sd)
          auc.df <- cbind(auc.df, row)
      }
        return(auc.df)
      }
      df <- GetMinMaxMeanSd(bidder.subset.aucs[,1:4])
      print(count.time)
      count.time <<- count.time - 1
      print(df)
      bidder.time <<- rbind(bidder.time, df)
}

tapply(bidder.train.ext.3$bidder_id, bidder.train.ext.3$bidder_id, GetTimeDiff)
rownames(bidder.time) <- NULL

# apply the same extraction process to the testing data set
bidder.time.test <- data.frame()
count.time.test <- nrow(bidder.test)
GetTimeDiffTest <- function(bidder.id) {
  # the function is used in conjuction with tapply through either bids.full or bids.w.test
  # it iterates through each bidder_id, finds all associated auctions for that bidder,
  # first finds min, mean, max, and sd of all time differences between each bid in a given auction,
  # second it calculates the min, mean, max, and sd of the above values for any given auction, across
  # all auctions.
  # Eventually the function will update the bidder.time data.frame to include 16 time related features.
  #
  # Args:
  #   bidder.id: the bidder_id factor; to be used as a key to iterate through in tapply
  #
  # returns: nothing;
  # but through each iteration, bidder.time is updated;
  # bidder.`time will have the following features:
  # mean.min.diff.auc, max.min.diff.auc, min.min.diff.auc, sd.min.diff.auc,
  # mean.mean.diff.auc, max.mean.diff.auc, min.mean.diff.auc, sd.mean.diff.auc,
  # mean.max.diff.auc, max.max.diff.auc, min.max.diff.auc, sd.max.diff.auc,
  # mean.sd.diff.auc, max.sd.diff.auc, min.sd.diff.auc, sd.sd.diff.auc
  bidder.subset <- subset(bids.w.test, bidder_id == bidder.id)
  bidder.subset <- bidder.subset[order(bidder.subset$time), ]
  bidder.subset.aucs <- data.frame()
  lapply(split(bidder.subset, as.factor(as.character(bidder.subset$auction))), function(auc.subset) {
    # the lapply iterates through each of the auction subset,
    # and extracts min, mean, max, and sd of bid time difference for each auction
    times <- sort(auc.subset$time)
    if (length(times) > 1) {
      new.times <- times[2:length(times)]
    } else {
      # if only one bid is found for an auction, get time from the first bid of next auction
      # if only one bid is found for an user, or there is no subsequent bids, 
      #  get next bid time from the max value of the dataset
      new.times <- bidder.subset[grep(auc.subset$auction[1], bidder.subset$auction) + 1, 'time'] 
      if (is.na(new.times)) {
        new.times <- max(bids.full$time)
      }
      
    }
    diff <- new.times - times[1:(length(times) - 1)]
    sd <- ifelse(is.na(sd(diff)), 0, sd(diff) )
    row <- c(min(diff), mean(diff), max(diff), sd)
    
    
    # handle only 1 bid in the auction
    
    row <- data.frame(t(data.frame(row)))
    row[5] <- as.character(auc.subset$auction[1])
    row[6] <- as.character(bidder.id)
    names(row) <- c("min.diff.auc", "mean.diff.auc", "max.diff.auc", "sd.diff.auc", "auction", "bidder_id")
    bidder.subset.aucs <<- rbind(bidder.subset.aucs, row)
  })
  # if only one bid is found, fill in the min, mean, max with mean of time difference across all auctions
  # if sd.diff.auc is mean for an auction, fill in 0
#   mean.all <- mean(bidder.subset.aucs[!is.na(bidder.subset.aucs$mean.diff.auc), 'mean.diff.auc'])
#   bidder.subset.aucs[is.na(bidder.subset.aucs[,1]), 1] <- mean.all
#   bidder.subset.aucs[is.na(bidder.subset.aucs[,2]), 2] <- mean.all
#   bidder.subset.aucs[is.na(bidder.subset.aucs[,3]), 3] <- mean.all
#   bidder.subset.aucs[is.na(bidder.subset.aucs[,4]), 4] <- 0
  # now extract the second layer of information: the distribution of the summary metrics
  GetMinMaxMeanSd <- function(df) {
  # a function to extract min, max, mean, and sd from each column of a data.frame
  #  Args:
  #   df: a data.frame
  #   return a one-row data.frame with min, max, mean, and sd for each original column feature
    auc.df <- data.frame(bidder_id = bidder.id)
    for (i in 1:(ncol(df)-1)) {
    min <- min(as.numeric(df[,i]))
    max <- max(as.numeric(df[,i]))
    mean <- mean(as.numeric(df[,i]))
    sd <- sd(as.numeric(df[,i]))
    row <- c(min, max, mean, sd)
    row <- data.frame(t(data.frame(row)))
    feature <- names(df)[i]
    name.min <- paste0("min", ".", feature)
    name.mean <- paste0("mean", ".", feature)
    name.max <- paste0("max", ".", feature)
    name.sd <- paste0("sd", ".", feature)
    names(row) <- c(name.min, name.mean, name.max, name.sd)
    auc.df <- cbind(auc.df, row)
  }
    return(auc.df)
}
  df <- GetMinMaxMeanSd(bidder.subset.aucs[,1:4])
  print(count.time.test)
  count.time.test <<- count.time.test - 1
  print(df)
  bidder.time.test <<- rbind(bidder.time.test, df)
}

tapply(bidder.test.ext.3$bidder_id, bidder.test.ext.3$bidder_id, GetTimeDiffTest)
rownames(bidder.time.test) <- NULL


# merge bidder.time.1 and bidder.time.test.1 to 
bidder.train.ext.4 <- merge(bidder.train.ext.3, bidder.time)
bidder.test.ext.4 <- merge(bidder.test.ext.3, bidder.time.test)

bidder.test.ext.4[is.na(bidder.test.ext.4$sd.min.diff.auc), 'sd.min.diff.auc'] <- 0
bidder.test.ext.4[is.na(bidder.test.ext.4$sd.mean.diff.auc), 'sd.mean.diff.auc'] <- 0
bidder.test.ext.4[is.na(bidder.test.ext.4$sd.max.diff.auc), 'sd.max.diff.auc'] <- 0






# remove redundant features
bidder.train.ext.4 <- bidder.train.ext.4[, !names(bidder.train.ext.4) %in% 
                                           c('payment_account', 'address', 'bids.min',
                                             'devices.min', 'countries.min', 'ips.min',
                                             'urls.min')]
bidder.test.ext.4 <- bidder.test.ext.4[, !names(bidder.test.ext.4) %in% 
                                           c('payment_account', 'address', 'bids.min',
                                             'devices.min', 'countries.min', 'ips.min',
                                             'urls.min')]
# save(bidder.train.ext.4, file = "bidder.train.ext.4.RData", ascii = T, compress = T)
# save(bidder.test.ext.4, file = "bidder.test.ext.4.RData", ascii = T, compress = T)
load("bidder.train.ext.4.RData")
load("bidder.test.ext.4.RData")




# select only the most important features from both models
rf.import <- varImp(rf.model.full)$importance
gbm.import <- varImp(gbm.model.full)$importance

rf.import$feature <- rownames(rf.import)
gbm.import$feature <- rownames(gbm.import)

rownames(rf.import) <- NULL
rownames(gbm.import) <- NULL

rf.import <- rf.import[order(-rf.import$Overall), ]
gbm.import <- gbm.import[order(-gbm.import$Overall), ]

names(rf.import) <- c("rf.overall", "feature")
names(gbm.import) <- c("gbm.overall", "feature")

# take the most important 25 features from each model, select only the overlap
merged.import <- merge(rf.import[1:25,], gbm.import[1:25, ])
merged.import <- merged.import[order(-merged.import$rf.overall), ]

# save file so running rf and gbm model to get varImp is not needed every time
# save(merged.import, file = "merged.import.RData")
load("merged.import.RData")

# extract country feature
country.code <- read.csv("country_code.csv")

bidder.country <- data.frame()
count.country <- nrow(bids.full)
lapply(split(bids.full, as.factor(as.character(bids.full$bidder_id))), function(x) {
  country.bids.bidder.part <- data.frame(country = country.code$code)
  df <- aggregate(bidder_id ~ country, x, length)
  sum <- sum(df$bidder_id)
  country.bids.bidder.part <- merge(country.bids.bidder.part, df, all = T)
  country.bids.bidder.part[is.na(country.bids.bidder.part$bidder_id), 2] <- 0
  country.bids.bidder.part$bidder_id <- round(country.bids.bidder.part$bidder_id * 100 / sum, 2)
  country.bids.bidder.part.t <- t(country.bids.bidder.part [, 2])
  colnames(country.bids.bidder.part.t) <- as.character(t(country.bids.bidder.part[, 1]))
  country.bids.bidder.part.t <- data.frame(country.bids.bidder.part.t)
  rownames(country.bids.bidder.part.t) <- NULL
  
  country.bids.bidder.part.t$bidder_id <- as.character(x$bidder_id[1])
  bidder.country <<- rbind(bidder.country, country.bids.bidder.part.t)
  count.country <<- count.country - 1
  print(count.country)
  
})

bidder.train.ext.5 <- merge(bidder.train.ext.4, bidder.country)


# extract country feature for test set
bidder.country.test <- data.frame()
count.country.test <- nrow(bidder.test.ext.4)
lapply(split(bids.w.test, as.factor(as.character(bids.w.test$bidder_id))), function(x) {
  country.bids.bidder.part <- data.frame(country = country.code$code)
  df <- aggregate(bidder_id ~ country, x, length)
  sum <- sum(df$bidder_id)
  country.bids.bidder.part <- merge(country.bids.bidder.part, df, all = T)
  country.bids.bidder.part[is.na(country.bids.bidder.part$bidder_id), 2] <- 0
  country.bids.bidder.part$bidder_id <- round(country.bids.bidder.part$bidder_id * 100 / sum, 2)
  country.bids.bidder.part.t <- t(country.bids.bidder.part [, 2])
  colnames(country.bids.bidder.part.t) <- as.character(t(country.bids.bidder.part[, 1]))
  country.bids.bidder.part.t <- data.frame(country.bids.bidder.part.t)
  rownames(country.bids.bidder.part.t) <- NULL
  
  country.bids.bidder.part.t$bidder_id <- as.character(x$bidder_id[1])
  bidder.country.test <<- rbind(bidder.country.test, country.bids.bidder.part.t)
  count.country.test <<- count.country.test - 1
  print(count.country.test)
  
})

bidder.test.ext.5 <- merge(bidder.test.ext.4, bidder.country.test)

# save(bidder.train.ext.5, file = 'bidder.train.ext.5.RData')
# save(bidder.test.ext.5, file = 'bidder.test.ext.5.RData')
load('bidder.train.ext.5.RData')
load('bidder.test.ext.5.RData')

# add quantile features
# for training set
bidder.train.ext.6 <- data.frame(bidder_id = bidder.train.ext.4$bidder_id, 
                                 outcome = bidder.train.ext.4$outcome)
feats.to.quant <- names(bidder.train.ext.4[, !names(bidder.train.ext.4) 
                                           %in% c('bidder_id', 'outcome', 'merchandise')])
for (i in 1:length(feats.to.quant)) {
  quant.name <- paste0(feats.to.quant[i], ".quant")
  quant <- (bidder.train.ext.4[, feats.to.quant[i]] - 
              mean(bidder.train.ext.4[, feats.to.quant[i]])) / sd(bidder.train.ext.4[, feats.to.quant[i]])
  quant <- pnorm(quant)
  bidder.train.ext.6[, quant.name] <- quant          
            
}

# for testing set
bidder.test.ext.6 <- data.frame(bidder_id = bidder.test.ext.4$bidder_id, 
                                 outcome = bidder.test.ext.4$outcome)
feats.to.quant <- names(bidder.test.ext.4[, !names(bidder.test.ext.4) 
                                           %in% c('bidder_id', 'outcome', 'merchandise')])
for (i in 1:length(feats.to.quant)) {
  quant.name <- paste0(feats.to.quant[i], ".quant")
  quant <- (bidder.test.ext.4[, feats.to.quant[i]] - 
              mean(bidder.test.ext.4[, feats.to.quant[i]])) / sd(bidder.test.ext.4[, feats.to.quant[i]])
  quant <- pnorm(quant)
  bidder.test.ext.6[, quant.name] <- quant          
  
}

# are they useful? fit a RF to find out
rf.quant <- train(as.factor(outcome) ~ . - bidder_id, bidder.train.ext.6,
                                 method = 'rf', metric = 'ROC',
                                 trControl = trainControl(method = 'cv', 
                                                          number = 10,
                                                          classProbs = T,
                                                          summaryFunction = twoClassSummary))


rf.original <- train(as.factor(outcome) ~ . - bidder_id - merchandise, bidder.train.ext.4,
                  method = 'rf', metric = 'ROC',
                  trControl = trainControl(method = 'cv', 
                                           number = 10,
                                           classProbs = T,
                                           summaryFunction = twoClassSummary))

# rf.quant has sligtly better CV scores; keep it

# bidder.country info for train and test
bidder.country <- bidder.train.ext.5[, 37:290]
bidder.country.test <- bidder.test.ext.5[, 37:290]

bidder.train.ext.6 <- merge(bidder.train.ext.6, bidder.country)
bidder.test.ext.6 <- merge(bidder.test.ext.6, bidder.country.test)
bidder.train.ext.6$merchandise <- bidder.train.ext.4$merchandise
bidder.test.ext.6$merchandise <- bidder.test.ext.4$merchandise

save(bidder.train.ext.6, file = 'bidder.train.ext.6.Rdata')
save(bidder.test.ext.6, file = 'bidder.test.ext.6.RData')
load('bidder.train.ext.6.Rdata')
load('bidder.test.ext.6.RData')

bidder.train.ext.7 <- merge(bidder.train.ext.6, bidder.train.ext.5)
bidder.test.ext.7 <- merge(bidder.test.ext.6, bidder.test.ext.5)


# save(bidder.train.ext.7, file = 'bidder.train.ext.7.Rdata')
# save(bidder.test.ext.7, file = 'bidder.test.ext.7.RData')











