# first, the model that scored 0.89 on leaderboard
head(bidder.train.ext.3)
head(bidder.test.ext.3)


# models
gbmGrid <- expand.grid(interaction.depth = c(1, 5, 9),
                       n.trees = (1:30)*50,
                       shrinkage = 0.1,
                       n.minobsinnode = 10)

gbm.model.no.time <- train(as.factor(outcome) ~ . - bidder_id - payment_account - address - merchandise, 
                        bidder.train.ext.3, method = 'gbm', metric = 'ROC',
                        verbose=T, 
                        tuneGrid = gbmGrid,
                        trControl = trainControl(method = 'cv', 
                                                 number = 10,
                                                 classProbs = T,
                                                 summaryFunction = twoClassSummary))

gbmGrid <- expand.grid(interaction.depth = 1,
                       n.trees = 300,
                       shrinkage = 0.1,
                       n.minobsinnode = 10)

gbm.model.no.time.1 <- train(as.factor(outcome) ~ . - bidder_id - payment_account - address - merchandise, 
                           bidder.train.ext.3, method = 'gbm', metric = 'ROC',
                           verbose=T, 
                           tuneGrid = gbmGrid,
                           trControl = trainControl(method = 'cv', 
                                                    number = 10,
                                                    classProbs = T,
                                                    summaryFunction = twoClassSummary))


rfGrid <-  expand.grid(mtry = c(2:31))
rf.model.no.time <- train(as.factor(outcome) ~ . - bidder_id - payment_account - address - merchandise, 
                          bidder.train.ext.3, method = 'rf', metric = 'ROC',
                          tuneGrid = rfGrid,
                          trControl = trainControl(method = 'cv',
                                                   classProbs = T,
                                                   summaryFunction = twoClassSummary))

# hand pick model based on graph, based on gbm.model.no.time
rfGrid <-  expand.grid(mtry = 2)
rf.model.no.time.1 <- train(as.factor(outcome) ~ . - bidder_id - payment_account - address - merchandise, 
                          bidder.train.ext.3, method = 'rf', metric = 'ROC',
                          tuneGrid = rfGrid,
                          trControl = trainControl(method = 'cv',
                                                   classProbs = T,
                                                   summaryFunction = twoClassSummary))




# gbm.model, best model by caret: n.trees = 150, depth = 5
auc.gbm.no.time <- CrossValidation(gbm.model.no.time, bidder.train.ext.3)
# gbm.model, hand picked from graph: n.trees = 300, depth = 1
auc.gbm.no.time.1 <- CrossValidation(gbm.model.no.time.1, bidder.train.ext.3)

# rf model, best model by caret: mtry = 6
auc.rf.no.time.no.merc <- CrossValidation(rf.model.no.time, bidder.train.ext.3)
# rf model, hand picked from graph: mtry = 2
auc.rf.no.time.no.merc.1 <- CrossValidation(rf.model.no.time.1, bidder.train.ext.3)

# rf full model, best model by caret: mtry = 14
auc.rf.full <- CrossValidation(rf.model.full, bidder.train.ext.4)
# gbm full model, best model by caret: ntrees = 300
auc.gbm.full <- auc.gbm.full.ntrees.300

# rf hand picked model from rf.full.model, use mtry = 2
rfGrid <-  expand.grid(mtry = 2)
rf.model.full.mtry.2 <- train(as.factor(outcome) ~ . - bidder_id - payment_account - address, 
                       bidder.train.ext.4, method = 'rf', metric = 'ROC',
                       tuneGrid = rfGrid,
                       trControl = trainControl(method = 'cv', 
                                                number = 10,
                                                classProbs = T,
                                                summaryFunction = twoClassSummary))
auc.rf.full.mtry.2 <- CrossValidation(rf.model.full.mtry.2, bidder.train.ext.4)

# rf top 20 features only, mtry = 5 by caret
auc.rf.full.2 <- CrossValidation(rf.model.full.2, bidder.train.ext.4)

# rf top 20 features only, mtry = 11 to recreate leaderboard
rfGrid <-  expand.grid(mtry = 11)
rf.model.shared.mtry.11 <- train(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% merged.import$feature], 
                                 y =  as.factor(bidder.train.ext.4$outcome),
                                 method = 'rf', metric = 'ROC',
                                 tuneGrid = rfGrid,
                                 trControl = trainControl(method = 'cv', 
                                                          number = 10,
                                                          classProbs = T,
                                                          summaryFunction = twoClassSummary))
auc.rf.shared.mtry.11 <- CrossValidation(rf.model.shared.mtry.11, bidder.train.ext.4)

# rf top 20 features only, mtry = 3 to recreate leaderboard
rfGrid <-  expand.grid(mtry = 3)
rf.model.shared.mtry.3 <- train(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% merged.import$feature], 
                                y =  as.factor(bidder.train.ext.4$outcome),
                                method = 'rf', metric = 'ROC',
                                tuneGrid = rfGrid,
                                trControl = trainControl(method = 'cv', 
                                                         number = 10,
                                                         classProbs = T,
                                                         summaryFunction = twoClassSummary))

auc.rf.shared.mtry.3 <- CrossValidation(rf.model.shared.mtry.3, bidder.train.ext.4)

# gbm top features only, n.trees = 550, depth = 1
auc.gbm.full.2 <- CrossValidation(gbm.model.full.2, bidder.train.ext.4)

# rf no time no merc, but only add max.max.diff.auc (caret picked mtry = 9)
bidder.train.ext.5 <- bidder.train.ext.3
bidder.train.ext.5$max.max.diff.auc <- bidder.train.ext.4$max.max.diff.auc
rfGrid <-  expand.grid(mtry = (2:10))
rf.model.one.time <- train(as.factor(outcome) ~ . - bidder_id - merchandise - payment_account - address,
                           bidder.train.ext.5,
                                 method = 'rf', metric = 'ROC',
                                 tuneGrid = rfGrid,
                                 trControl = trainControl(method = 'cv', 
                                                          number = 10,
                                                          classProbs = T,
                                                          summaryFunction = twoClassSummary))
auc.rf.model.one.time.mtry.9 <- CrossValidation(rf.model.one.time, bidder.train.ext.4)

# rf no time no mer, only add max.max.diff.auc, mtry = 4
rfGrid <-  expand.grid(mtry = 4)
rf.model.one.time.mtry.4 <- train(as.factor(outcome) ~ . - bidder_id - merchandise - payment_account - address,
                           bidder.train.ext.5,
                           method = 'rf', metric = 'ROC',
                           tuneGrid = rfGrid,
                           trControl = trainControl(method = 'cv', 
                                                    number = 10,
                                                    classProbs = T,
                                                    summaryFunction = twoClassSummary))
auc.rf.model.one.time.mtry.4 <- CrossValidation(rf.model.one.time.mtry.4, bidder.train.ext.4)

# delete bids.min, merchandise, devices.min, countries.min, ips.min, urls.min, payment_acocunt,
# and address
bidder.train.ext.4 <- bidder.train.ext.4[, !names(bidder.train.ext.4) %in% 
                                           c('bids.min', 'merchandise', 'devices.min', 'countries.min',
                                             'urls.min', 'payment_account', 'address')]
bidder.test.ext.4 <- bidder.test.ext.4[, !names(bidder.test.ext.4) %in% 
                                         c('bids.min', 'merchandise', 'devices.min', 'countries.min',
                                           'urls.min', 'payment_account', 'address')]

# rf and gbm model with top 15 features from merged.import, mtry = 4 by caret
rfGrid <-  expand.grid(mtry = c(2:10))
features <- merged.import$feature[1:15]
rf.top.15.features <- train(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% features], 
                         y =  as.factor(bidder.train.ext.4$outcome),
                         method = 'rf', metric = 'ROC',
                         tuneGrid = rfGrid,
                         trControl = trainControl(method = 'cv',
                                                  classProbs = T,
                                                  summaryFunction = twoClassSummary))
auc.rf.top.15.feat.mtry.4 <- CrossValidation(rf.top.15.features, bidder.train.ext.4)

gbmGrid <- expand.grid(interaction.depth = c(1, 5, 9),
                       n.trees = (1:30)*50,
                       shrinkage = 0.1,
                       n.minobsinnode = 10)
gbm.top.15.features <- train(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% merged.import$feature], 
                          y =  as.factor(bidder.train.ext.4$outcome),
                          method = 'gbm', metric = 'ROC',
                          verbose=T,
                          tuneGrid = gbmGrid,
                          trControl = trainControl(method = 'cv', 
                                                   number = 10,
                                                   classProbs = T,
                                                   summaryFunction = twoClassSummary))
# gbm modeol, n.trees = 100, depth = 5
auc.gbm.top.15.feat <- CrossValidation(gbm.top.15.features, bidder.train.ext.4)

# tune a customized randomForest
rf.tune.top.15 <- randomForest(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% features], 
                               y =  as.factor(bidder.train.ext.4$outcome),
                               ntree = 300,
                               mtry = 4)

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
  for (j in 1:10) {
    
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
  } 
  
  return(auc) 
  
}

auc.rf.tune.ntree.150 <- CrossValidationRF(ntree = 150, df = bidder.train.ext.4, 
                                           features = features)
auc.rf.tune.ntree.200 <- CrossValidationRF(ntree = 200, df = bidder.train.ext.4, 
                                           features = features)
auc.rf.tune.ntree.250 <- CrossValidationRF(ntree = 250, df = bidder.train.ext.4, 
                                           features = features)
auc.rf.tune.ntree.250.mtry.3 <- CrossValidationRF(ntree = 250, mtry = 3, df = bidder.train.ext.4, 
                                           features = features)
auc.rf.tune.ntree.250.mtry.5 <- CrossValidationRF(ntree = 250, mtry = 5, df = bidder.train.ext.4, 
                                                  features = features)
auc.rf.tune.ntree.250.mtry.5.feat.10 <- CrossValidationRF(ntree = 250, mtry = 5, df = bidder.train.ext.4, 
                                                  features = features[1:10])
auc.rf.tune.ntree.250.mtry.3.feat.10 <- CrossValidationRF(ntree = 250, mtry = 3, df = bidder.train.ext.4, 
                                                          features = features[1:5])


auc.rf.tune.ntree.250.mtry.5.classwt.1.5 <- CrossValidationRF(ntree = 250, mtry = 5, classwt = c(1, 1.5),
                                                  df = bidder.train.ext.4, features = features)
auc.rf.tune.ntree.250.mtry.5.classwt.1.1 <- CrossValidationRF(ntree = 250, mtry = 5, classwt = c(1, 1.1),
                                                              df = bidder.train.ext.4, features = features)
auc.rf.tune.ntree.250.mtry.6 <- CrossValidationRF(ntree = 250, mtry = 6, df = bidder.train.ext.4, 
                                                  features = features)
auc.rf.tune.ntree.250.mtry.7 <- CrossValidationRF(ntree = 250, mtry = 7, df = bidder.train.ext.4, 
                                                  features = features)
auc.rf.tune.ntree.250.mtry.15 <- CrossValidationRF(ntree = 250, mtry = 15, df = bidder.train.ext.4, 
                                                  features = features)
auc.rf.tune.ntree.300 <- CrossValidationRF(ntree = 300, df = bidder.train.ext.4, 
                                           features = features)
auc.rf.tune.ntree.350 <- CrossValidationRF(ntree = 350, df = bidder.train.ext.4, 
                                           features = features)
auc.rf.tune.ntree.1000 <- CrossValidationRF(ntree = 1000, df = bidder.train.ext.4, 
                                            features = features)
auc.rf.tune.ntree.1500 <- CrossValidationRF(ntree = 1500, df = bidder.train.ext.4, 
                                            features = features)
auc.rf.tune.ntree.2000 <- CrossValidationRF(ntree = 2000, df = bidder.train.ext.4, 
                                           features = features)

auc.rf.tune.ntree.1500.feat.10 <- CrossValidationRF(ntree = 1500, df = bidder.train.ext.4,
                                                    features = features[1:10])


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
  row <- c(mean, median, round(score, 5), round(sd, 3))
  row <- data.frame(t(data.frame(row)))
  row$model.desc <- descr
  names(row) <- c("mean", "median", "leaderboard", "sd", "model.desc")
  model.comp <<- rbind(model.comp, row)
  
}
AddModelCV(auc.gbm.no.time, "gbm, no time, ntrees 150, depth 5", 0.85864)
AddModelCV(auc.gbm.no.time.1, "gbm, no time, ntrees 300, depth 1") 
AddModelCV(auc.rf.no.time.no.merc, "rf, no time and merc, mtry = 6")
AddModelCV(auc.rf.no.time.no.merc.1, "rf model, no time and merc, mtry = 2", 0.89195)
AddModelCV(auc.rf.full, "rf full model all features, mtry = 14 by caret", 0.88006)
AddModelCV(auc.rf.full.mtry.2, "rf full model bu mtry = 2", 0.87986)
AddModelCV(auc.gbm.full, "gbm full model, ntrees = 300 by caret", 0.87715)
AddModelCV(auc.rf.full.2, "rf, top 20 shared features, mtry =5 by caret", 0.88643) # good performance
AddModelCV(auc.gbm.full.2, "gbm, top 20 shared features, ntrees 550, depth 1")
AddModelCV(auc.rf.shared.mtry.11, "rf, top 20 overlap features, mtry 11", 0.87956)
AddModelCV(auc.rf.shared.mtry.3, "rf, top 20 overlap features, mtry 3", 0.87834)
AddModelCV(auc.rf.model.one.time.mtry.9, 'rf, no time no merc, only max.max.diff; mtry 9')
AddModelCV(auc.rf.model.one.time.mtry.4, 'rf, no time no merc, only max.max.diff; mtry 4', 0.86902)
AddModelCV(auc.rf.top.15.feat.mtry.4, 'rf, top 15 overlapped features; mtry 4')
AddModelCV(auc.rf.tune.ntree.150, 'rf, ntree 150, mtry 4, top 15 feats')
AddModelCV(auc.rf.tune.ntree.200, 'rf, ntree 200, mtry 4, top 15 feats')
AddModelCV(auc.rf.tune.ntree.250, 'rf, ntree 250, mtry 4, top 15 feats')
AddModelCV(auc.rf.tune.ntree.250.mtry.3, 'rf, ntree 250, mtry 4, top 15 feats')
AddModelCV(auc.rf.tune.ntree.250.mtry.5, 'rf, ntree 250, mtry 5, top 15 feats', 0.87874)
AddModelCV(auc.rf.tune.ntree.250.mtry.5.classwt.1.5, 'rf, ntree 250, mtry 5, top 15 feats, classwt 1.5')
AddModelCV(auc.rf.tune.ntree.250.mtry.5.classwt.1.1, 'rf, ntree 250, mtry 5, top 15 feats, classwt 1.1')
AddModelCV(auc.rf.tune.ntree.250.mtry.5.feat.10, 'rf, ntree 250, mtry 5, top 10 feats')
AddModelCV(auc.rf.tune.ntree.250.mtry.6, 'rf, ntree 250, mtry 6, top 15 feats')
AddModelCV(auc.rf.tune.ntree.250.mtry.7, 'rf, ntree 250, mtry 7, top 15 feats')
AddModelCV(auc.rf.tune.ntree.250.mtry.15, 'rf, ntree 250, mtry 15, top 15 feats')
AddModelCV(auc.rf.tune.ntree.300, 'rf, ntree 300, mtry 4, top 15 feats')
AddModelCV(auc.rf.tune.ntree.350, 'rf, ntree 350, mtry 4, top 15 feats')
AddModelCV(auc.rf.tune.ntree.1000, 'rf, ntree 1000, mtry 4, top 15 feats')
AddModelCV(auc.rf.tune.ntree.1500, 'rf, ntree 1500, mtry 4, top 15 feats')
AddModelCV(auc.rf.tune.ntree.2000, 'rf, ntree 2000, mtry 4, top 15 feats')

rownames(model.comp) <- NULL
model.comp <- model.comp[order(-model.comp$mean),]
# save(model.comp, file = "model.comp.RData")
load("model.comp.RData")

leaderboard.cv <- subset(model.comp, leaderboard != 0)
leaderboard.cv$p <- pnorm((leaderboard.cv$leaderboard - leaderboard.cv$mean) / (leaderboard.cv$sd))
leaderboard.cv
















