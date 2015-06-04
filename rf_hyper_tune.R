# hyper parameter tuning for caret Random Forest

rfGrid <- expand.grid(mtry = c(2:10))
rf.full.country.25.feat <- train(x = bidder.train.ext.5[, names(bidder.train.ext.5) %in% rf.full.features$feature[1:25]], 
                                 y = as.factor(bidder.train.ext.5$outcome), 
                                 method = 'rf', metric = 'ROC',
                                 tuneGrid = rfGrid,
                                 trControl = trainControl(method = 'cv', 
                                                          number = 10,
                                                          classProbs = T,
                                                          summaryFunction = twoClassSummary))


auc.rf.country.df <- data.frame()
counter <- 55
for (i in 4:58) {
  num.feats <- i * 5
  tune.feats <- rf.full.features$feature[1:num.feats]
  rfGrid <- expand.grid(mtry = floor(sqrt(num.feats)))
  model <- train(x = bidder.train.ext.5[, names(bidder.train.ext.5) %in% tune.feats], 
                 y = as.factor(bidder.train.ext.5$outcome), 
                 method = 'rf', metric = 'ROC',
                 tuneGrid = rfGrid,
                 trControl = trainControl(method = 'cv', 
                                          number = 10,
                                          classProbs = T,
                                          summaryFunction = twoClassSummary))
  best.mtry <- model$finalModel$mtry
  auc.model <- CrossValidation(model, bidder.train.ext.5)
  mean.auc.model <- mean(auc.model)
  median.auc.model <- median(auc.model)
  sd.auc.model <- sd(auc.model)
  auc.model.row <- c(mean.auc.model, median.auc.model, sd.auc.model, round(best.mtry, 0), round(num.feats, 0))
  auc.rf.country.df <<- rbind(auc.rf.country.df, auc.model.row)
  names(auc.rf.country.df) <- c('mean', 'median', 'sd', 'mtry', 'n.feats')
  counter <<- counter - 1
  print(counter)

}

auc.rf.country.df <- auc.rf.country.df[order(-auc.rf.country.df$mean), ]

# based on previous leaderboard data, predict the score on leaderboard
auc.rf.country.df$pred <- auc.rf.country.df$mean + qnorm(mean(leaderboard.cv$p)) * auc.rf.country.df$sd



# hyper tune with cforest ntree
auc.crf.country.df <- data.frame()
counter <- 30
for (i in 10:40) {
  num.trees <- i * 50
  
  # modify the CV function to accomodate cforest
  CrossValidationCRF <- function(df, ntree, mtry, num.feats) {
    # this function inputs creates a randomForest model based on arguments, 
    # checks AUC with a repeated 10-fold validation
    # and returns the AUC value for all repeated 10 folds
    #
    # Args:
    #   df: the data.frame to be used for cross-validation
    #   ntree: number of trees to grow in the randomForest model
    #   mtry: number of variables at each split
    #   num.feats: number of features, selected from varImp produced by caret
    #   
    # returns: 
    #   a list with AUC value for each fold 
    
    auc <- c()
    
    tune.feats <- rf.full.features$feature[1:num.feats]
    features <- c(tune.feats, 'merchandise', 'outcome')
    # repeated the 10-fold cross validation 10 times
    
    folds <- createFolds(y = df$outcome, k = 10, list = T, returnTrain = F)  
    for (i in 1:length(folds)) {
      
      inTest <- folds[[i]]
      intest.fold <- df[inTest, ]
      intrain.fold <- df[-inTest, ]
      # replace X0 and X1 with 0 and 1
      #   bidder.train[bidder.train$outcome == 1, "outcome"] <- "X1"
      #   bidder.train[bidder.train$outcome == 0, "outcome"] <- "X0"
      #   
      intest.fold[intest.fold$outcome == "X0", "outcome"] <- 0
      intest.fold[intest.fold$outcome == "X1", "outcome"] <- 1
      
      intest.fold$outcome <- as.numeric(intest.fold$outcome)
      
      y <- as.factor(intrain.fold$outcome)
      # subset intest.fold to only include related features
      intest.fold <- intest.fold[, names(intest.fold) %in% features]
      
      
      intrain <- intrain.fold[, names(intrain.fold) %in% features]
  
      intrain.model <- cforest(as.factor(outcome) ~ . ,
                               data = intrain[, names(intrain) %in% features],
                               control = cforest_unbiased(ntree = ntree, 
                                                          mtry = mtry))
      
      pred.fold <- predict(intrain.model, intest.fold, OOB = T, type = "prob")
      unlist.pred <- unlist(pred.fold)
      pred.x1 <- c()
      for (i in 1:(length(unlist.pred) / 2)) {
        pred.x1 <- c(pred.x1, unlist.pred[i * 2])
      }
      roc.fold <- prediction(pred.x1, as.numeric(intest.fold$outcome))
      auc.fold <- performance(roc.fold, measure = 'auc')@y.values[[1]]
      print(auc.fold)      
      auc <- c(auc, auc.fold)       
      
    }
    
    return(auc) 
    
  }
  
  auc.model <- CrossValidationCRF(df = bidder.train.ext.5,
                                  ntree = num.trees,
                                  mtry = 17,
                                  num.feats = 289)
  
  mean.auc.model <- mean(auc.model)
  median.auc.model <- median(auc.model)
  sd.auc.model <- sd(auc.model)
  auc.model.row <- c(mean.auc.model, median.auc.model, sd.auc.model, 17, num.trees)
  auc.crf.country.df <<- rbind(auc.crf.country.df, auc.model.row)
  names(auc.crf.country.df) <- c('mean', 'median', 'sd', 'mtry', 'ntrees')
  counter <<- counter - 1
  print(counter)
  
}
auc.crf.country.df <- auc.crf.country.df[order(-auc.crf.country.df$mean), ]

# hyper tune with caret, features fixed at 140
rfGrid <- expand.grid(mtry = c(2:60))
rf.feat.140 <- train(x = bidder.train.ext.5[, names(bidder.train.ext.5) %in% rf.full.features$feature[1:140]], 
                                 y = as.factor(bidder.train.ext.5$outcome), 
                                 method = 'rf', metric = 'ROC',
                                 tuneGrid = rfGrid,
                                 trControl = trainControl(method = 'cv', 
                                                          number = 10,
                                                          classProbs = T,
                                                          summaryFunction = twoClassSummary))
auc.rf.feat.140 <- CrossValidation(rf.feat.140, bidder.train.ext.5)

# hyper tune with cforest ntree for feats fiexed at 140 and mtry at 37
auc.crf.country.df.140 <- data.frame()
counter <- 30
for (i in 10:40) {
  num.trees <- i * 50
  
  # modify the CV function to accomodate cforest
  CrossValidationCRF <- function(df, ntree, mtry, num.feats) {
    # this function inputs creates a randomForest model based on arguments, 
    # checks AUC with a repeated 10-fold validation
    # and returns the AUC value for all repeated 10 folds
    #
    # Args:
    #   df: the data.frame to be used for cross-validation
    #   ntree: number of trees to grow in the randomForest model
    #   mtry: number of variables at each split
    #   num.feats: number of features, selected from varImp produced by caret
    #   
    # returns: 
    #   a list with AUC value for each fold 
    
    auc <- c()
    
    tune.feats <- rf.full.features$feature[1:num.feats]
    features <- c(tune.feats, 'merchandise', 'outcome')
    # repeated the 10-fold cross validation 10 times
    
    folds <- createFolds(y = df$outcome, k = 10, list = T, returnTrain = F)  
    for (i in 1:length(folds)) {
      
      inTest <- folds[[i]]
      intest.fold <- df[inTest, ]
      intrain.fold <- df[-inTest, ]
      # replace X0 and X1 with 0 and 1
      #   bidder.train[bidder.train$outcome == 1, "outcome"] <- "X1"
      #   bidder.train[bidder.train$outcome == 0, "outcome"] <- "X0"
      #   
      intest.fold[intest.fold$outcome == "X0", "outcome"] <- 0
      intest.fold[intest.fold$outcome == "X1", "outcome"] <- 1
      
      intest.fold$outcome <- as.numeric(intest.fold$outcome)
      
      y <- as.factor(intrain.fold$outcome)
      # subset intest.fold to only include related features
      intest.fold <- intest.fold[, names(intest.fold) %in% features]
      
      
      intrain <- intrain.fold[, names(intrain.fold) %in% features]
      
      intrain.model <- cforest(as.factor(outcome) ~ . ,
                               data = intrain[, names(intrain) %in% features],
                               control = cforest_unbiased(ntree = ntree, 
                                                          mtry = mtry))
      
      pred.fold <- predict(intrain.model, intest.fold, OOB = T, type = "prob")
      unlist.pred <- unlist(pred.fold)
      pred.x1 <- c()
      for (i in 1:(length(unlist.pred) / 2)) {
        pred.x1 <- c(pred.x1, unlist.pred[i * 2])
      }
      roc.fold <- prediction(pred.x1, as.numeric(intest.fold$outcome))
      auc.fold <- performance(roc.fold, measure = 'auc')@y.values[[1]]
      print(auc.fold)      
      auc <- c(auc, auc.fold)       
      
    }
    
    return(auc) 
    
  }
  
  auc.model <- CrossValidationCRF(df = bidder.train.ext.5,
                                  ntree = num.trees,
                                  mtry = 37,
                                  num.feats = 140)
  
  mean.auc.model <- mean(auc.model)
  median.auc.model <- median(auc.model)
  sd.auc.model <- sd(auc.model)
  auc.model.row <- c(mean.auc.model, median.auc.model, sd.auc.model, 17, num.trees, num.feats)
  auc.crf.country.df.140 <<- rbind(auc.crf.country.df.140, auc.model.row)
  names(auc.crf.country.df.140) <- c('mean', 'median', 'sd', 'mtry', 'ntrees', 'feats')
  counter <<- counter - 1
  print(counter)
  
}
auc.crf.country.df.140 <- auc.crf.country.df.140[order(-auc.crf.country.df.140$mean), ]


# see bidder.train.ext.6 performance compared to bidder.train.ext.5 with RF; using doPar for repeated 10 fold CV
cl <- makeCluster(10)
registerDoParallel(cl)
rf.caret.tune.quant <- foreach(i = 1:10, .packages = c('caret', 'ROCR'), .combine = rbind) %dopar% {
  rfGrid <-  expand.grid(mtry = c(4, 11, 17))
  rf.model.full.quant.caret <- train(as.factor(outcome) ~ . - bidder_id, 
                                bidder.train.ext.6, method = 'rf', metric = 'ROC',
                                tuneGrid = rfGrid,
                                trControl = trainControl(method = 'cv', 
                                                         number = 10,
                                                         classProbs = T,
                                                         summaryFunction = twoClassSummary))
  rf.model.full.quant.caret$results  
    
}

save(rf.caret.tune.quant, file = 'rf.caret.tune.quant.RData')
rf.caret.tune.quant <- aggregate(cbind(ROC, ROCSD) ~ mtry, rf.caret.tune.quant, mean)
rf.caret.tune.quant <- rf.caret.tune.quant[order(-rf.caret.tune.quant$ROC), ]



# closing cluster
stopCluster(cl)
registerDoSEQ()





