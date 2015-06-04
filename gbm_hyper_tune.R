# use GBM boosting to further increase performance of RF
# select tuning parameters that yielded the best results in the RF models in model.comp
gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 4, 11, 17),
                        shrinkage = c(0.001, 0.01),
                        n.trees = c(1:100) * 50,
                        n.minobsinnode = c(10, 15, 20))
gbm.model.full <- train(as.factor(outcome) ~ . - bidder_id, 
                        bidder.train.ext.5, method = 'gbm', metric = 'ROC',
                        trControl = trainControl(method = 'cv', 
                                                 number = 10,
                                                 classProbs = T,
                                                 summaryFunction = twoClassSummary))


# hyper tune gbm; relying on loop-based CV instead of Caret, which overestimates performance

# create a data.frame containing all the parameter combinations
# each row represents a set of tuning parameters
depth <- c(1, 2, 4, 11, 17)
shrinkage <- c(0.001, 0.01)
ntree <- seq(1000, 5000, 100)
gbm.params <- expand.grid(depth = depth, 
                          shrinkage = shrinkage, 
                          ntree = ntree)
# hyper tune with doParallel; self CV
library(doParallel)
cl <- makeCluster(16)
registerDoParallel(cl)
# for (i in 1:3)
auc.gbm.tune.df <- foreach(i = 1:10, .packages = c('caret', 'ROCR'), .combine = rbind) %dopar% {
  # iterate through each row, take the set of parameters, calculate its AUC
  CrossValidationGBM <- function(df, depth, shrinkage, ntree, n.minobsinnode) {
    # this function creates a gbm model based on arguments, 
    # checks AUC with a repeated 10-fold validation
    # and returns the AUC value for all repeated 10 folds
    #
    # Args:
    #   df: the data.frame to be used for cross-validation
    #   depth: number of splits
    #   shrinkage: the shrinkage value to be added for each boost
    #   ntree: number of trees to iterating the boosting in the gbm model
    #
    # returns: 
    #   a list with AUC value for each fold 
    
    auc <- c()
    
#     tune.feats <- rf.full.features$feature[1:num.feats]
#     features <- c(tune.feats, 'merchandise', 'outcome')
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
#       intest.fold <- intest.fold[, names(intest.fold) %in% features]
      
      
      intrain <- intrain.fold
      
#       intrain.model <- cforest(as.factor(outcome) ~ . ,
#                                data = intrain[, names(intrain) %in% features],
#                                control = cforest_unbiased(ntree = ntree, 
#                                                           mtry = mtry))
#      
      gbmGrid <-  expand.grid(interaction.depth = depth,
                              shrinkage = shrinkage,
                              n.trees = ntree,
                              n.minobsinnode = 20)
      intrain.model <- train(as.factor(outcome) ~ . - bidder_id, 
                              intrain, method = 'gbm', metric = 'ROC',
                              tuneGrid = gbmGrid,
                              trControl = trainControl(method = 'none',
                                                       classProbs = T,
                                                       summaryFunction = twoClassSummary))
      
          
    pred.fold <- predict(intrain.model, intest.fold, type = "prob")
    roc.fold <- prediction(pred.fold$X1, intest.fold$outcome)
    auc.fold <- performance(roc.fold, measure = 'auc')@y.values[[1]]
    print(auc.fold)      
    auc <- c(auc, auc.fold)   

    }
    
    return(auc) 
    
  }
  
  auc.model <- CrossValidationGBM(df = bidder.train.ext.5,
                                  ntree = gbm.params[i, 'ntree'],
                                  depth = gbm.params[i, 'depth'],
                                  shrinkage = gbm.params[i, 'shrinkage'],
                                  n.minobsinnode = 20)
  
  mean.auc.model <- mean(auc.model)
  median.auc.model <- median(auc.model)
  sd.auc.model <- sd(auc.model)
  auc.model.row <- c(mean.auc.model, median.auc.model, sd.auc.model, 
                     gbm.params[i, 'ntree'], 
                     gbm.params[i, 'depth'],
                     gbm.params[i, 'shrinkage'],
                     20)
  auc.model.row
#   auc.gbm.tune.df <<- rbind(auc.gbm.tune.df, auc.model.row)
#   names(auc.gbm.tune.df) <- c('mean', 'median', 'sd', 'ntree', 'depth', 'shrinkage', 'minobs')
#   counter <<- counter - 1
#   print(counter)
  
}
auc.gbm.tune.df <- data.frame(auc.gbm.tune.df)
names(auc.gbm.tune.df) <- c('mean', 'median', 'sd', 'ntree', 'depth', 'shrinkage', 'minobs')
auc.gbm.tune.df <- auc.gbm.tune.df[order(-auc.gbm.tune.df$mean), ]
save(auc.gbm.tune.df, file = 'auc.gbm.tune.df.RData')
stopCluster(cl)
registerDoSEQ()

auc.gbm.tune.df.full <- rbind(auc.gbm.tune.df.full, auc.gbm.tune.df)
auc.gbm.tune.df.full <- auc.gbm.tune.df.full[order(-auc.gbm.tune.df.full$mean), ]



# hyper tune with doParallel; Caret -- 10 x 10 CV
cl <- makeCluster(10)
registerDoParallel(cl)
gbm.caret.tune <- foreach(i = 1:10, .packages = c('caret', 'ROCR'), .combine = rbind) %dopar% {
    gbmGrid <-  expand.grid(interaction.depth = c(1, 4, 11, 17),
                            shrinkage = c(0.001, 0.01),
                            n.trees = c(1:100) * 50,
                            n.minobsinnode = c(10, 20))
    gbm.model.full.caret <- train(as.factor(outcome) ~ . - bidder_id, 
                            bidder.train.ext.5, method = 'gbm', metric = 'ROC',
                            tuneGrid = gbmGrid,
                            trControl = trainControl(method = 'cv', 
                                                     number = 10,
                                                     classProbs = T,
                                                     summaryFunction = twoClassSummary))
    gbm.model.full.caret$results  
    
  
}
save(gbm.caret.tune, file = 'gbm.caret.tune.RData')

stopCluster(cl)
registerDoSEQ()

gbm.caret.tune.clean <- aggregate(cbind(ROC, ROCSD) ~ shrinkage + interaction.depth + n.minobsinnode + n.trees,
                                 gbm.caret.tune, mean)
gbm.caret.tune.clean <- gbm.caret.tune.clean[order(-gbm.caret.tune.clean$ROC), ]
save(gbm.caret.tune.clean, file = 'gbm.caret.tune.clean.RData')


# see performance with GBM caret without hyper tune, on bidder.train.ext.6 data








# # hyper tune with doParallel on bidder.train.ext.6, quantile data -- 10 x 10 CV
# cl <- makeCluster(10)
# registerDoParallel(cl)
# gbm.caret.tune.quant <- foreach(i = 1:10, .packages = c('caret', 'ROCR'), .combine = rbind) %dopar% {
#   gbmGrid <-  expand.grid(interaction.depth = c(1, 4, 11, 17),
#                           shrinkage = c(0.001, 0.01),
#                           n.trees = c(1:100) * 50,
#                           n.minobsinnode = c(10, 20))
#   gbm.model.full.caret <- train(as.factor(outcome) ~ . - bidder_id, 
#                                 bidder.train.ext.6, method = 'gbm', metric = 'ROC',
#                                 tuneGrid = gbmGrid,
#                                 trControl = trainControl(method = 'cv', 
#                                                          number = 10,
#                                                          classProbs = T,
#                                                          summaryFunction = twoClassSummary))
#   gbm.model.full.caret$results  
#     
# }
# save(gbm.caret.tune.quant, file = 'gbm.caret.tune.quant.RData')
# 
# 
# 
# 
# 
# 











