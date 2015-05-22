# apply the best performing model to the testing set
 
# submission 14
bidder.test.ext.5 <- bidder.test.ext.3
bidder.test.ext.5$max.max.diff.auc <- bidder.test.ext.4$max.max.diff.auc
# rfGrid <-  expand.grid(mtry = 4)
# rf.model.one.time.mtry.4 <- train(as.factor(outcome) ~ . - bidder_id - merchandise - payment_account - address,
#                                   bidder.train.ext.5,
#                                   method = 'rf', metric = 'ROC',
#                                   tuneGrid = rfGrid,
#                                   trControl = trainControl(method = 'cv', 
#                                                            number = 10,
#                                                            classProbs = T,
#                                                            summaryFunction = twoClassSummary))

pred.test <- predict(rf.model.one.time.mtry.4, bidder.test.ext.5,
                     type = "prob")
submit <- data.frame(bidder_id = bidder.test.ext.5$bidder_id, prediction = pred.test$X1)
# merge the original test data bidder_id and submit
submit.full <- merge(submit, test, all=T)
submit.full[is.na(submit.full$prediction), "prediction"] <- 0
submit.full <- submit.full[, 1:2]
write.csv(submit.full, 'submit_14.csv', row.names = F)


# submission 15
features <- merged.import$feature[1:15]
# auc.rf.tune.ntree.250.mtry.5 <- CrossValidationRF(ntree = 250, mtry = 5, df = bidder.train.ext.4, 
#                                                   features = features)
rf.tune.ntree.250.mtry.5 <- randomForest(x = bidder.train.ext.4[, names(bidder.train.ext.4) %in% features],
                                         y = as.factor(bidder.train.ext.4$outcome),
                                         ntree = 250, mtry = 5)
pred.test <- predict(rf.tune.ntree.250.mtry.5, bidder.test.ext.4[, names(bidder.test.ext.4) %in% features],
                     type = "prob")

submit <- data.frame(bidder_id = bidder.test.ext.4$bidder_id, prediction = pred.test[, "X1"])

# merge the original test data bidder_id and submit
submit.full <- merge(submit, test, all=T)
submit.full[is.na(submit.full$prediction), "prediction"] <- 0
submit.full <- submit.full[, 1:2]
write.csv(submit.full, 'submit_15.csv', row.names = F)
