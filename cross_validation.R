# create 10-cross validation with the final models
CrossValidation <- function(model, df) {
  # this function inputs a model, checks AUC with a repeated 10-fold validation
  # and returns the AUC value for all repeated 10 folds
  #
  # Args:
  #   model: the model to be evaluated
  #     
  # returns: 
  #   a list with AUC value for each fold 
  features <- c(model$finalModel$xNames, "outcome")
 
  folds <- createFolds(y = df$outcome, k = 10, list = T, returnTrain = F)
  
  # get model method
  method <- model$method
  
  # get model tuneValues
  tune.values <- model$finalModel$tuneValue
  rownames(tune.values) <- NULL
  tuneGrid <- expand.grid(tune.values)
  
  # trControl with no validation method
  trControl = trainControl(method = 'none',
                           classProbs = T,
                           summaryFunction = twoClassSummary)
  
  
  auc <- c()
  
  # 10-fold cross validation

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
    
    # subset intest.fold to only include related features
    intest.fold <- intest.fold[, names(intest.fold) %in% features]
    
    intrain <- intrain.fold[, names(intrain.fold) %in% features]
    y <- as.factor(intrain$outcome)
    
    intrain.model <- train(x = intrain[, names(intrain) != 'outcome'],
                           y = y,
                           method = method,
                           tuneGrid = tuneGrid,
                           trControl = trControl)
    
    pred.fold <- predict(intrain.model, intest.fold, type = "prob")
    roc.fold <- prediction(pred.fold$X1, intest.fold$outcome)
    auc.fold <- performance(roc.fold, measure = 'auc')@y.values[[1]]
    print(auc.fold)      
    auc <- c(auc, auc.fold)      
    
  }

  
  return(auc) 
  
}

# fitting models

# fit a RF model with all observations from the bidder.train dataset, use caret to examine CV 
# and tuning parameters

rfGrid <-  expand.grid(mtry = c(2:35))
rf.model.full <- train(as.factor(outcome) ~ . - bidder_id, 
                       bidder.train.ext.4, method = 'rf', metric = 'ROC',
                       tuneGrid = rfGrid,
                       trControl = trainControl(method = 'cv',
                                                number = 10,
                                                classProbs = T,
                                                summaryFunction = twoClassSummary))

# rf full model, best model by caret: mtry = 16
auc.rf.full <- CrossValidation(rf.model.full, bidder.train.ext.4)

rfGrid <-  expand.grid(mtry = 14)
rf.model.full.14 <- train(as.factor(outcome) ~ . - bidder_id, 
                       bidder.train.ext.4, method = 'rf', metric = 'ROC',
                       tuneGrid = rfGrid,
                       trControl = trainControl(method = 'none',
                                                classProbs = T,
                                                summaryFunction = twoClassSummary))

# rf full model, best model by caret: mtry = 14
auc.rf.full.14 <- CrossValidation(rf.model.full.14, bidder.train.ext.4)

# find the most important features according to caret RF
rf.import <- data.frame(rf.model.full$finalModel$importance)
rf.import$var <- rownames(rf.import)
rownames(rf.import) <- NULL
rf.import <- rf.import[order(-rf.import$MeanDecreaseGini), ]
# save(rf.import, file = "rf.import.RData")
load('rf.import.RData')

# fit the models again, with only selected top 20 features by rf
rfGrid <-  expand.grid(mtry = c(2:20))
rf.20.feats <- train(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% rf.import$var[1:20]], 
                         y =  as.factor(bidder.train.ext.4$outcome),
                         method = 'rf', metric = 'ROC',
                         tuneGrid = rfGrid,
                         trControl = trainControl(method = 'cv',
                                                  number = 10,
                                                  classProbs = T,
                                                  summaryFunction = twoClassSummary))
auc.20.feats <- CrossValidation(rf.20.feats, bidder.train.ext.4)
# fit the models again, with only selected top 20 features by merged.import
rfGrid <-  expand.grid(mtry = c(2:20))
rf.20.feats.merged <- train(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% merged.import$feature], 
                     y =  as.factor(bidder.train.ext.4$outcome),
                     method = 'rf', metric = 'ROC',
                     tuneGrid = rfGrid,
                     trControl = trainControl(method = 'cv', 
                                              number = 10,
                                              classProbs = T,
                                              summaryFunction = twoClassSummary))
auc.20.feats.merged <- CrossValidation(rf.20.feats.merged, bidder.train.ext.4)



# benchmark: leaderboard score recreation

# hand pick model based on graph, based on gbm.model.no.time
rfGrid <-  expand.grid(mtry = 2)
rf.model.no.time.1 <- train(as.factor(outcome) ~ . - bidder_id - payment_account - address - merchandise, 
                          bidder.train.ext.3, method = 'rf', metric = 'ROC',
                          tuneGrid = rfGrid,
                          trControl = trainControl(method = 'none',
                                                   classProbs = T,
                                                   summaryFunction = twoClassSummary))
# rf model, hand picked from graph: mtry = 2
auc.rf.no.time.no.merc.1 <- CrossValidation(rf.model.no.time.1, bidder.train.ext.3)


# rf hand picked model from rf.full.model, use mtry = 2
rfGrid <-  expand.grid(mtry = 2)
rf.model.full.mtry.2 <- train(as.factor(outcome) ~ . - bidder_id,
                       bidder.train.ext.4, method = 'rf', metric = 'ROC',
                       tuneGrid = rfGrid,
                       trControl = trainControl(method = 'none',
                                                classProbs = T,
                                                summaryFunction = twoClassSummary))
auc.rf.full.mtry.2 <- CrossValidation(rf.model.full.mtry.2, bidder.train.ext.4)

# rf top 20 features only, mtry = 11 to recreate leaderboard
rfGrid <-  expand.grid(mtry = 11)
rf.model.shared.mtry.11 <- train(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% merged.import$feature], 
                                 y =  as.factor(bidder.train.ext.4$outcome),
                                 method = 'rf', metric = 'ROC',
                                 tuneGrid = rfGrid,
                                 trControl = trainControl(method = 'none', 
                                                          classProbs = T,
                                                          summaryFunction = twoClassSummary))
auc.rf.shared.mtry.11 <- CrossValidation(rf.model.shared.mtry.11, bidder.train.ext.4)

# rf top 20 features only, mtry = 3 to recreate leaderboard
rfGrid <-  expand.grid(mtry = 3)
rf.model.shared.mtry.3 <- train(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% merged.import$feature], 
                                y =  as.factor(bidder.train.ext.4$outcome),
                                method = 'rf', metric = 'ROC',
                                tuneGrid = rfGrid,
                                trControl = trainControl(method = 'none', 
                                                         classProbs = T,
                                                         summaryFunction = twoClassSummary))

auc.rf.shared.mtry.3 <- CrossValidation(rf.model.shared.mtry.3, bidder.train.ext.4)



# rf no time no mer, only add max.max.diff.auc, mtry = 4
rfGrid <-  expand.grid(mtry = 4)
rf.model.one.time.mtry.4 <- train(as.factor(outcome) ~ . - bidder_id - merchandise,
                           bidder.train.ext.5,
                           method = 'rf', metric = 'ROC',
                           tuneGrid = rfGrid,
                           trControl = trainControl(method = 'none',
                                                    classProbs = T,
                                                    summaryFunction = twoClassSummary))
auc.rf.model.one.time.mtry.4 <- CrossValidation(rf.model.one.time.mtry.4, bidder.train.ext.4)

# # delete bids.min, merchandise, devices.min, countries.min, ips.min, urls.min, payment_acocunt,
# # and address
# bidder.train.ext.4 <- bidder.train.ext.4[, !names(bidder.train.ext.4) %in% 
#                                            c('bids.min', 'merchandise', 'devices.min', 'countries.min',
#                                              'urls.min', 'payment_account', 'address')]
# bidder.test.ext.4 <- bidder.test.ext.4[, !names(bidder.test.ext.4) %in% 
#                                          c('bids.min', 'merchandise', 'devices.min', 'countries.min',
#                                            'urls.min', 'payment_account', 'address')]

# rf and gbm model with top 15 features from merged.import, mtry = 4 by caret
rfGrid <-  expand.grid(mtry = c(2:10))
features <- merged.import$feature[1:15]
rf.top.15.features <- train(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% features], 
                         y =  as.factor(bidder.train.ext.4$outcome),
                         method = 'rf', metric = 'ROC',
                         tuneGrid = rfGrid,
                         trControl = trainControl(method = 'cv',
                                                  number = 10,
                                                  classProbs = T,
                                                  summaryFunction = twoClassSummary))
auc.rf.top.15.feat.mtry.4 <- CrossValidation(rf.top.15.features, bidder.train.ext.4)


# rf on country features
# added > 200 country features; use RF to select the most important ones
rfGrid <-  expand.grid(mtry = 17)
rf.full.country <- train(as.factor(outcome) ~ . - bidder_id, 
                         bidder.train.ext.5, method = 'rf', metric = 'ROC',
                         tuneGrid = rfGrid,
                         trControl = trainControl(method = 'none', 
                                                  classProbs = T,
                                                  summaryFunction = twoClassSummary))
auc.rf.country <- CrossValidation(rf.full.country, bidder.train.ext.5)

# rf country features, top 25 only, mtry 4
rfGrid <- expand.grid(mtry = 4)
rf.full.country.25.feat.mtry.4 <- train(x = bidder.train.ext.5[, names(bidder.train.ext.5) %in% rf.full.features$feature[1:25]], 
                                        y = as.factor(bidder.train.ext.5$outcome), 
                                        method = 'rf', metric = 'ROC',
                                        tuneGrid = rfGrid,
                                        trControl = trainControl(method = 'none',
                                                                 classProbs = T,
                                                                 summaryFunction = twoClassSummary))
auc.rf.country.25.feat.mtry.4 <- CrossValidation(rf.full.country.25.feat.mtry.4, bidder.train.ext.5)

# rf on country features - select the most important 30
rf.full.features <- rf.full.country$finalModel$importance
rf.full.features <- data.frame(rf.full.features)
rf.full.features$feature <- rownames(rf.full.features)
rf.full.features <- rf.full.features[order(-rf.full.features[, 1]), ]
rownames(rf.full.features) <- NULL

rfGrid <-  expand.grid(mtry = c(2:10))
rf.full.country.25.feat <- train(x = bidder.train.ext.5[, names(bidder.train.ext.5) %in% rf.full.features$feature[1:25]], 
                         y = as.factor(bidder.train.ext.5$outcome), 
                         method = 'rf', metric = 'ROC',
                         tuneGrid = rfGrid,
                         trControl = trainControl(method = 'cv', 
                                                  number = 10,
                                                  classProbs = T,
                                                  summaryFunction = twoClassSummary))

# select no. 3 from auc.rf.country.df
rfGrid <- expand.grid(mtry = 11)
rf.full.country.140.feat.mtry.11 <- train(x = bidder.train.ext.5[, names(bidder.train.ext.5) %in% rf.full.features$feature[1:140]], 
                                          y = as.factor(bidder.train.ext.5$outcome), 
                                          method = 'rf', metric = 'ROC',
                                          tuneGrid = rfGrid,
                                          trControl = trainControl(method = 'none',
                                                                   classProbs = T,
                                                                   summaryFunction = twoClassSummary))
auc.rf.full.country.140.feat.mtry.11 <- CrossValidation(rf.full.country.140.feat.mtry.11, bidder.train.ext.5)







features <- merged.import$feature
# CrossValidation for randomForest
CrossValidationRF <- function(df, features, ntree = 500, mtry = 4,
                              replace = TRUE, classwt = NULL, importance = TRUE) {
  # this function inputs creates a randomForest model based on arguments, 
  # checks AUC with a repeated 10-fold validation
  # and returns the AUC value for all repeated 10 folds
  #
  # Args:
  #   df: the data.frame to be used for cross-validation
  #   features: features to include in the model
  #   ntree: number of trees to grow in the randomForest model
  #   mtry: number of variables at each split
  #   replace: if TRUE, sample selection is replaced
  #   classwt: weight given to each class
  #   importance: if TRUE, importance for each variable is calculated
  #     
  # returns: 
  #   a list with AUC value for each fold 
  
  auc <- c()
  
  features <- c(features, 'outcome')
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
      
      intrain.model <- randomForest(x = intrain[, names(intrain) != 'outcome'],
                             y = y,
                             ntree = ntree, mtry = mtry,
                             replace = replace, classwt = classwt, 
                             importance = importance)
      
      pred.fold <- predict(intrain.model, intest.fold, type = "prob")
      roc.fold <- prediction(pred.fold[, 'X1'], as.numeric(intest.fold$outcome))
      auc.fold <- performance(roc.fold, measure = 'auc')@y.values[[1]]
      print(auc.fold)      
      auc <- c(auc, auc.fold)       
      
    }
  
  
  return(auc) 
  
}

auc.rf.tune.ntree.250.mtry.5 <- CrossValidationRF(ntree = 250, mtry = 5, df = bidder.train.ext.4, 
                                                  features = features)

auc.rf.tune.ntree.250.mtry.5.classwt.1.1 <- CrossValidationRF(ntree = 250, mtry = 5, classwt = c(1, 1.1),
                                                              df = bidder.train.ext.4, features = features)

# mtry 6, n feats 40, highest mean according to CV
rfGrid <- expand.grid(mtry = 6)
rf.full.country.40.feat.mtry.6 <- train(x = bidder.train.ext.5[, names(bidder.train.ext.5) %in% rf.full.features$feature[1:40]], 
                                        y = as.factor(bidder.train.ext.5$outcome), 
                                        method = 'rf', metric = 'ROC',
                                        tuneGrid = rfGrid,
                                        trControl = trainControl(method = 'none',
                                                                 classProbs = T,
                                                                 summaryFunction = twoClassSummary))
auc.rf.full.country.40.feat.mtry.6 <- CrossValidation(rf.full.country.40.feat.mtry.6, bidder.train.ext.5)


# gbm models - see gbm_hyper_tune.R
# select tuning parameters that yielded the best results in the RF models in model.comp
# gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 4, 6, 8, 17),
#                         shrinkage = c(0.001, 0.01, 0.1),
#                         n.trees = c(1:100) * 50,
#                         n.minobsinnode = 20)
# gbm.model.full <- train(as.factor(outcome) ~ . - bidder_id, 
#                         bidder.train.ext.5, method = 'gbm', metric = 'ROC',
#                         tuneGrid = gbmGrid,
#                         trControl = trainControl(method = 'cv', 
#                                                  number = 10,
#                                                  classProbs = T,
#                                                  summaryFunction = twoClassSummary))
auc.gbm.full <- CrossValidation(gbm.model.full, bidder.train.ext.5)

# select tuning parameters that yielded one of the best results but as a simpler model:
# depth 4, ntrees 3950, shrinkage 0.01, with ROC 0.9196007 in Caret and 0.03531 SD, one of the lowest in top 25
gbmGrid <-  expand.grid(interaction.depth = 4,
                        shrinkage = 0.01,
                        n.trees = 3950,
                        n.minobsinnode = 20)
gbm.model.full.depth.4 <- train(as.factor(outcome) ~ . - bidder_id, 
                        bidder.train.ext.5, method = 'gbm', metric = 'ROC',
                        tuneGrid = gbmGrid,
                        trControl = trainControl(method = 'none',
                                                 classProbs = T,
                                                 summaryFunction = twoClassSummary))
auc.gbm.full.depth.4 <- CrossValidation(gbm.model.full.depth.4, bidder.train.ext.5)


# depth 17, ntrees 1800, shrinkage 0.001, with ROC 0.9237138 in CV (first)
gbmGrid <-  expand.grid(interaction.depth = 17,
                        shrinkage = 0.001,
                        n.trees = 1800,
                        n.minobsinnode = 20)
gbm.model.full.depth.17.ntree.1800 <- train(as.factor(outcome) ~ . - bidder_id, 
                                            bidder.train.ext.5, method = 'gbm', metric = 'ROC',
                                            tuneGrid = gbmGrid,
                                            trControl = trainControl(method = 'none',
                                                                     classProbs = T,
                                                                     summaryFunction = twoClassSummary))
auc.gbm.full.depth.17.ntree.1800 <- CrossValidation(gbm.model.full.depth.17.ntree.1800, bidder.train.ext.5)

# depth 17, ntrees 1100, shrinkage 0.001, with ROC 0.9213210 in CV (first)
gbmGrid <-  expand.grid(interaction.depth = 17,
                        shrinkage = 0.001,
                        n.trees = 1100,
                        n.minobsinnode = 20)
gbm.model.full.depth.17.ntree.1100 <- train(as.factor(outcome) ~ . - bidder_id, 
                                            bidder.train.ext.5, method = 'gbm', metric = 'ROC',
                                            tuneGrid = gbmGrid,
                                            trControl = trainControl(method = 'none',
                                                                     classProbs = T,
                                                                     summaryFunction = twoClassSummary))
auc.gbm.full.depth.17.ntree.1100 <- CrossValidation(gbm.model.full.depth.17.ntree.1100, bidder.train.ext.5)

# depth 4, ntrees 1400, shrinkage 0.01, with ROC 0.9188608 in CV (first)
gbmGrid <-  expand.grid(interaction.depth = 4,
                        shrinkage = 0.01,
                        n.trees = 1400,
                        n.minobsinnode = 20)
gbm.model.full.depth.4.ntree.1400 <- train(as.factor(outcome) ~ . - bidder_id, 
                                            bidder.train.ext.5, method = 'gbm', metric = 'ROC',
                                            tuneGrid = gbmGrid,
                                            trControl = trainControl(method = 'none',
                                                                     classProbs = T,
                                                                     summaryFunction = twoClassSummary))
auc.gbm.full.depth.4.ntree.1400 <- CrossValidation(gbm.model.full.depth.4.ntree.1400, bidder.train.ext.5)

# depth 17, ntrees 1400, shrinkage 0.001, with ROC 0.9222540 in CV (first)
gbmGrid <-  expand.grid(interaction.depth = 17,
                        shrinkage = 0.001,
                        n.trees = 1400,
                        n.minobsinnode = 20)
gbm.model.full.depth.17.ntree.1400 <- train(as.factor(outcome) ~ . - bidder_id, 
                                           bidder.train.ext.5, method = 'gbm', metric = 'ROC',
                                           tuneGrid = gbmGrid,
                                           trControl = trainControl(method = 'none',
                                                                    classProbs = T,
                                                                    summaryFunction = twoClassSummary))
auc.gbm.full.depth.17.ntree.1400 <- CrossValidation(gbm.model.full.depth.4.ntree.1400, bidder.train.ext.5)




# create a data.frame to compare model CV performances
model.comp <- data.frame(mean = as.numeric(),
                         median = as.numeric(),
                         leaderboard = as.numeric(),                         
                         sd = as.numeric(),
                         model.desc = as.character())                         
AddModelCV <- function(auc, descr, score = 0) {
  mean <- mean(auc)
  median <- median(auc)
  sd <- sd(auc)
  p.value <- round(t.test(auc, mu = score)$p.value, 2)
  row <- c(mean, median, round(score, 5), round(sd, 3), p.value)
  row <- data.frame(t(data.frame(row)))
  row$model.desc <- descr
  
  names(row) <- c("mean", "median", "leaderboard", "sd", "p.value", "model.desc")
  
  model.comp <<- rbind(model.comp, row)
  
}

AddModelCV(auc.rf.no.time.no.merc.1, "rf model, no time and merc, mtry = 2", 0.89195)
AddModelCV(auc.rf.full.14, "rf full model all features, mtry = 14 by caret", 0.88006)
AddModelCV(auc.rf.full, "rf full model all features, mtry = 17 by caret", 0.87022)
AddModelCV(auc.rf.full.mtry.2, "rf full model bu mtry = 2", 0.87986)
# AddModelCV(auc.gbm.full, "gbm full model, ntrees = 300 by caret", 0.87715)
AddModelCV(auc.20.feats, "rf, top 20 rf features, mtry =5 by caret", 0.88643) 
AddModelCV(auc.rf.shared.mtry.11, "rf, top 20 overlap features, mtry 11", 0.87956)
AddModelCV(auc.rf.shared.mtry.3, "rf, top 20 overlap features, mtry 3", 0.87834)
AddModelCV(auc.rf.model.one.time.mtry.4, 'rf, no time no merc, only max.max.diff; mtry 4', 0.86902)
AddModelCV(auc.rf.top.15.feat.mtry.4, 'rf, top 15 overlapped features; mtry 4', 0.86105)
AddModelCV(auc.rf.tune.ntree.250.mtry.5, 'rf, ntree 250, mtry 5, top 15 feats', 0.87874)
AddModelCV(auc.rf.tune.ntree.250.mtry.5.classwt.1.1, 'rf, ntree 250, mtry 5, top 15 feats, classwt 1.1', 0.86337)
AddModelCV(auc.rf.country, 'rf, ntree 500, mtry 17, all features including country', 0.90415)
AddModelCV(auc.rf.country.25.feat.mtry.4, 'rf, ntree 500, mtry 4, top 25 features', 0.88378)
AddModelCV(auc.rf.full.country.40.feat.mtry.6, 'rf, ntree 500, mtry 6, top 40 features', 0.89266)
AddModelCV(auc.rf.full.country.140.feat.mtry.11, 'rf, ntree 500, mtry 11, top 140 features', 0.90323)
AddModelCV(auc.gbm.full, 'gbm, ntree 2100, depth 17, minobs 20, shrinkage 0.001 - ROC 0.9205843 with caret', 0.89083)
AddModelCV(auc.gbm.full.depth.4, 'gbm, ntree 3950, depth 4, minobs 20, shrinkage 0.01 - ROC 0.9196007 with Caret', 0.85817)
AddModelCV(auc.gbm.full.depth.17.ntree.1800, 'gbm, ntree 1800, depth 17, minobs 20, shrinkage 0.001',0.89393)
AddModelCV(auc.gbm.full.depth.17.ntree.1100, 'gbm, ntree 1100, depth 17, minobs 20, shrinkage 0.001')
AddModelCV(auc.gbm.full.depth.4.ntree.1400, 'gbm, ntree 1400, depth 4, minobs 20, shrinkage 0.01', 0.89130)

rownames(model.comp) <- NULL
model.comp <- model.comp[order(-model.comp$mean),]

save(model.comp, file = "model.comp.RData")
# load("model.comp.RData")








