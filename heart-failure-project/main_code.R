#####################################
# install and load required libraries
#####################################

if(!require(ggplot2))    install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggthemes))        install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(tidyverse))     install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))   install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot))     install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
if(!require(MASS))        install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(leaps))   install.packages("leaps", repos = "http://cran.us.r-project.org")
if(!require(rpart))      install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot))    install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(randomForest))        install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(e1071))     install.packages("e1071", repos = "http://cran.us.r-project.org")


library(ggplot2)
library(ggthemes)
library(tidyverse)
library(caret)
library(dslabs)
library(ggcorrplot)
library(MASS)
library(leaps)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)




#######################
# Import and clean data
#######################

# the github repo with the data set "heart_failure_clinical_records_dataset" is available here: (insert github link here) 
# the file "heart_failure_clinical_records_dataset.csv" provided in the github repo must be included in the working (project) directory for the code below to run

#read in data
heart <- read.csv("heart_failure_clinical_records_dataset.csv")

#inspect dataset
str(heart)

#rename DEATH_EVENT to "PD" (Patient Deceased)
heart <- heart %>% rename(PD = DEATH_EVENT)

#convert categorical variables into factors
heart$anaemia <- factor(heart$anaemia, levels = c(1,0))
levels(heart$anaemia) <- c("Yes", "No")
heart$diabetes <- factor(heart$diabetes, levels = c(1,0))
levels(heart$diabetes) <- c("Yes", "No")
heart$high_blood_pressure <- factor(heart$high_blood_pressure, levels = c(1,0))
levels(heart$high_blood_pressure) <- c("Yes", "No")
heart$sex <- factor(heart$sex, levels = c(1,0))
levels(heart$sex) <- c("Male", "Female")
heart$smoking <- factor(heart$smoking, levels = c(1,0))
levels(heart$smoking) <- c("Yes", "No")
heart$PD <- factor(heart$PD, levels = c(1,0))
levels(heart$PD) <- c("Yes", "No")

#The time feature refers to the time from the start of the study after which the study was terminated. 
#Either the people lost contact with the subject (presumably they were declared healthy and left) or they died due to heart failure. 
#Time therefore has an artificially high predictive power and should not be used to predict, as it would not be known in time to make a prediction
heart <- heart %>% dplyr::select(-time)


#create training set and validation dataset - 20% of total data.
#Valiation set will only be used to test our final chosen model

set.seed(1995, sample.kind = "Rounding")
validation_index <- createDataPartition(heart$PD, times = 1, p = 0.2, list = F)
validation <- heart %>%
  slice(validation_index)

train <- heart %>%
  slice(-validation_index)

#create train/test set from training data
#test set (20% of the training set) will be used to test our models prior to choosing a final model
set.seed(1992, sample.kind = "Rounding")
test_index <- createDataPartition(train$PD, times = 1, p = 0.2, list = F)
heart_train <- train %>%
  slice(-test_index)
heart_test <- train %>%
  slice(test_index)

###########################
# Exploratory Data Analysis
###########################
theme_set(theme_economist())


#see what proportion of cases lead to death
heart %>%
  group_by(PD) %>%
  summarise(proportion = n()/nrow(heart)) #about 32% of cases lead to death

#assess proportion of male/female cases
heart %>%
  group_by(sex) %>%
  summarise(proportion = n()/nrow(heart)) #65% male, 35% female

#assess proportion of death by gender
heart %>%
  group_by(sex, PD) %>%
  summarise(proportion = n()/nrow(heart)) #32% of women die, 32% of men die


#First we will assess the correlation between the features in the training set; we will create a correlation for both the factor variables
# and the continuous variables

#correlation matrix for factor variables
bin_heart <- sapply({train[,c(-1,-3,-5,-7,-8,-9,-10,-12)] == "Yes"} %>% as_tibble, as.numeric) %>% as_tibble %>% mutate(Gender = as.numeric(train$sex=="Male"))

#create correlation map: insignificant correlations are left blank:
cor_heart <- bin_heart %>% cor
p.cor_heart <- bin_heart %>% cor_pmat()
cor_heart %>%
  ggcorrplot(lab = TRUE, type = "lower", method = "circle",
             insig = "blank", p.mat = p.cor_heart,
             colors = c("#6D9EC1", "white", "#E46726"),
             title = "Correlation Plot of factor variables in Heart Failure Dataset", legend.title = "Correlation") #significant relation being gender and smoking, gender/diabetes and smoking/diabetes 
ggsave("images/correlation.png", width = 9, height = 9) 

#compare genders
train %>%
  ggplot(aes(PD, fill = sex)) +
  geom_bar(width = 0.6, position = position_dodge(width = 0.7)) +
  ylab("Number of deaths") +
  ggtitle("Number of Patient deaths by Gender") # death proportions are equal
ggsave("images/gender.png", width = 9, height = 9)

#age: heart failure is more likely as you get older, but slows down after around 60
train %>%
  ggplot(aes(age, PD, fill = PD)) +
  geom_violin(alpha = 0.8) +
  ggtitle("Distribution of deaths by Age")
ggsave("images/age.png", width = 9, height = 9)

#assess the gender/smoking correlation 
train %>%
  ggplot(aes(sex, smoking, colour = PD)) +
  geom_jitter(height = 0.2, width = 0.2) +
  ggtitle("Prevalence of patient deaths by Gender and Smoking") #women who smoke more likely to die (small sample however)
ggsave("images/gendersmoking.png", width = 9, height = 9)

#Assess the smoking/diabetes correlation 
train %>%
  ggplot(aes(diabetes, smoking, colour = PD)) +
  geom_jitter(height = 0.2, width = 0.2) +
  ggtitle("Prevalence of patient deaths by Diabetes and Smoking") #those with diabetes who smoke more likely to die
ggsave("images/smokingdiabetes.png", width = 9, height = 9)


#Gender/Diabetes - women with diabetes more likely to die than men with diabetes, but men more likely to die overall
train %>%
  ggplot(aes(sex, diabetes, colour = PD)) +
  geom_jitter(height = 0.2, width = 0.2) +
  ggtitle("Prevalence of patient deaths by Gender and Diabetes")
ggsave("images/genderdiabetes.png", width = 9, height = 9)

#Create correlation plot for the continuous variables:
cor(train[,c(1,3,5,7,8,9)]) %>%
  ggcorrplot(lab = TRUE, type = "lower", method = "circle",
             insig = "blank", p.mat = train[,c(1,3,5,7,8,9)] %>% cor_pmat(),
             colors = c("#6D9EC1", "white", "#E46726"),
             title = "Correlation plot of continuous variables in Heart Failure dataset", legend.title = "Correlation")
ggsave("images/correlation2.png", width = 9, height = 9)

#Look at creatinine phosphokinase levels of paitents - doesn't seem to be significant across genders or alive/dead patients
train %>%
  ggplot(aes(PD, creatinine_phosphokinase, fill = PD)) +
  geom_boxplot() +
  ggtitle("Distribution of patient deaths by creatinine phosphokinase")
ggsave("images/cp.png", width = 9, height = 9)


#ejection fraction - lower ejection fraction correlating to death in patients
#women on average have a higher EF
train %>%
  ggplot(aes(PD, ejection_fraction, fill = PD)) +
  geom_boxplot() +
  ggtitle("Distribution of patient deaths by ejection fraction")
ggsave("images/ef.png", width = 9, height = 9)

#platelets - doesn't seem to be significant
train %>%
  ggplot(aes(PD, platelets, fill = PD)) +
  geom_boxplot() +
  ggtitle("Distribution of patient deaths by platelet count")
ggsave("images/platelets.png", width = 9, height = 9)

#serium creatinine - more creatinine in the blood  is a factor towards death
train %>%
  ggplot(aes(PD, serum_creatinine, fill = PD)) +
  geom_boxplot() +
  ggtitle("Distribution of patient deaths by creatinine")
ggsave("images/sc.png", width = 9, height = 9)

#serium_sodium - less serum sodium in dead patients
train %>%
  ggplot(aes(PD, serum_sodium, fill = PD)) +
  geom_boxplot() +
  ggtitle("Distribution of patient deaths by sodium")
ggsave("images/ss.png", width = 9, height = 9)



#Serum sodium/ejection fraction relationship
train %>%
  ggplot(aes(ejection_fraction, serum_sodium, color = PD)) +
  geom_point() +
  ggtitle("Prevalence of patient deaths by ejection Fraction and sodium")
ggsave("images/ss_ef.png", width = 9, height = 9) #low ejection fraction seems to be a big factor in heart failure

#serum_sodium/serum creatnine relationship
train %>%
  ggplot(aes(serum_creatinine, serum_sodium, color = PD)) +
  geom_point() +
  ggtitle("Prevalence of patient deaths by sodium and creatinine")
ggsave("images/ss_sc.png", width = 9, height = 9) #serum creatinine > 0.125 seems to put people more at risk of heart failure

#serum creatinine/age relationship
train %>%
  ggplot(aes(age, serum_creatinine, color = PD)) +
  geom_point() +
  ggtitle("Prevalence of patient deaths by age and sodium")
ggsave("images/sc_age.png", width = 9, height = 9) #slight increase as you age,majority of people with levels <0.125 survive unless you're over 80


################################
#first model: logistic regression
################################

#Logistic Regression gives a probability a person will die upon next visit
#Accuracy and sensitivity are important metrics here as not identifying a high-risk patient may lead to death

# Logistic regression model predicts probabilties, the cutoff is chosen so that the mean of the accuracy and the sensitivity is maximised
#will try a range of cut-off values, p, as the cut off between Yes/No
#10-fold CV will be used to obtain the optimal p

p <- seq(0.0, 0.6, 0.01)
k <- 10
accuracy_p <- matrix(nrow = k, ncol = length(p))

#create folds
set.seed(2005, sample.kind = "Rounding")
ind <- createFolds(1:nrow(heart_train), k = k)

#perform cv
for (i in 1:k) {
  # create train and test sets for cv
  cv_train <- heart_train %>% slice(-ind[[i]])
  cv_test <- heart_train %>% slice(ind[[i]])
  
  # fill matrix with results (mean of accuracy and sensitivity) from glm with cutoff p
  accuracy_p[i,] <- sapply(p, function(p){
    
    # create the glm
    cv_mod_glm <- glm(as.numeric(PD =="Yes")~., family = binomial(logit), data = cv_train)
    
    # obtain the predictions (these are probabilities)
    cv_preds_glm <- predict(cv_mod_glm, cv_test, type = "response")
    
    # if predictions > p, class as Yes. 
    cv_cm <- confusionMatrix(ifelse(cv_preds_glm>p, "Yes","No") %>% factor(levels = c("Yes","No")), cv_test$PD)
    
    # return the mean of the accuracy and sensitivity
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
  })
  # keep track of how many seeds have been run
  cat("fold",i,"out of 10 complete\n")   
}

#extract optimal p
opt_p <- p[which(min_rank(desc(colMeans(accuracy_p)))==1)] 

#plot results from cv
tibble(p = p, mean_accuracy = colMeans(accuracy_p)) %>%
  ggplot(aes(p, mean_accuracy)) +
  geom_smooth()+
  geom_point() +
  geom_point(aes(opt_p, max(mean_accuracy)), shape = 5, size = 5) +
  xlab("Cutoff (p)") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various Cutoffs")
ggsave("images/cv_p.png", width = 8, height = 5)

#create model, look at significant variables
model_glm <- glm(as.numeric(PD == "Yes")~., family = binomial(logit), data = heart_train)
summary(model_glm)


preds_glm <- predict(model_glm, heart_test)
preds_glm <- ifelse(preds_glm > opt_p, "Yes", "No") %>% factor(levels = c("Yes","No"))
cm_glm <- confusionMatrix(preds_glm, heart_test$PD)


#try model again, but this time only with significant variables
rm(accuracy_p)
accuracy_p <- matrix(nrow = k, ncol = length(p))
for (i in 1:k) {
  # create train and test sets for cv
  cv_train <- heart_train %>% slice(-ind[[i]])
  cv_test <- heart_train %>% slice(ind[[i]])
  
  # fill matrix with results (mean of accuracy and sensitivity) from glm with cutoff p
  accuracy_p[i,] <- sapply(p, function(p){
    
    # create the glm
    cv_mod_glm <- glm(as.numeric(PD =="Yes")~age+ejection_fraction+serum_creatinine, family = binomial(logit), data = cv_train)
    
    # obtain the predictions (these are probabilities)
    cv_preds_glm <- predict(cv_mod_glm, cv_test, type = "response")
    
    # if prediction>p, class as Yes. 
    cv_cm <- confusionMatrix(ifelse(cv_preds_glm>p, "Yes","No") %>% factor(levels = c("Yes","No")), cv_test$PD)
    
    # return the mean of the accuracy and sensitivity
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
  })
  # keep track of how many seeds have been run
  cat("fold",i,"out of 10 complete\n") 
}

#extract optimal p and plot vs accuracy
opt_p_two <- p[which(min_rank(desc(colMeans(accuracy_p)))==1)] 

tibble(p = p, mean_accuracy = colMeans(accuracy_p)) %>%
  ggplot(aes(p, mean_accuracy)) +
  geom_smooth()+
  geom_point() +
  geom_point(aes(opt_p_two, max(mean_accuracy)), shape = 5, size = 5) +
  xlab("Cutoff (p)") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various Cutoffs")
ggsave("images/cv_p_2.png", width = 8, height = 5)

#Create model again
model_glm_two <- glm(as.numeric(PD == "Yes")~age+ejection_fraction+serum_creatinine, family = binomial(logit), data = heart_train)
preds_glm_two <- predict(model_glm_two, heart_test)
preds_glm_two <- ifelse(preds_glm_two > opt_p_two, "Yes", "No") %>% factor(levels = c("Yes","No"))

cm_glm_two <- confusionMatrix(preds_glm_two, heart_test$PD)

#store results
accuracy_results <- data.frame(Method = "Logistic Regression",
                           accuracy = cm_glm_two$overall["Accuracy"],
                           sensitivity = cm_glm_two$byClass["Sensitivity"])



#####################
#Naive Bayes 
#####################

#Create optimal model using the train() function and 10 fold cv
set.seed(1, sample.kind = "Rounding")
model_nb <- train(PD~.,
                  method = 'nb',
                  data = heart_train,
                  trControl = trainControl(method = 'cv', number = 10))

#Plot results
ggplot(model_nb)
ggsave("images/cv_nb.png", width = 8, height = 5)

#create predictions
preds_nb <- predict(model_nb, heart_test)

#confusion matrix
cm_nb <- confusionMatrix(preds_nb, heart_test$PD)

#store results
accuracy_results <- rbind(accuracy_results,
                          data.frame(Method = "Naive Bayes",
                                     accuracy = cm_nb$overall["Accuracy"],
                                     sensitivity = cm_nb$byClass["Sensitivity"]))


##############
#decision tree
##############

#will try a range of cp values for the ideal tree
#10-fold CV will be used to obtain the optimal cp

k <- 10
cp = seq(0,0.5,0.01)
accuracy_cp <- matrix(nrow = k, ncol = length(cp))

#create folds
set.seed(2005, sample.kind = "Rounding")
ind <- createFolds(1:nrow(heart_train), k = k)

#perform cv
for (i in 1:k) {
  # create train and test sets for cv
  cv_train <- heart_train %>% slice(-ind[[i]])
  cv_test <- heart_train %>% slice(ind[[i]])
  
# fill matrix with results (mean of accuracy and sensitivity) from decision tree 
  accuracy_cp[i,] <- sapply(cp, function(cp){
    
    # create the decision tree
    cv_mod_tree <- rpart(PD~., cp = cp, data =  cv_train, method = "class")
    
    # obtain the predictions 
    cv_preds_tree <- predict(cv_mod_tree, cv_test, type = "class")
    
    # confusion matrix 
    cv_cm <- confusionMatrix(cv_preds_tree, cv_test$PD)
    
    # return the mean of the accuracy and sensitivity
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
  })
  # keep track of how many seeds have been run
  cat("fold",i,"out of 10 complete\n") 
}

#extract optimal cp
opt_cp <- median(cp[which(min_rank(desc(colMeans(accuracy_cp)))==1)])

#plot cp vs accuracy
tibble(cp = cp, mean_accuracy = colMeans(accuracy_cp)) %>%
  ggplot(aes(cp, mean_accuracy)) +
  geom_smooth()+
  geom_point() +
  geom_point(aes(opt_cp, max(mean_accuracy)), shape = 5, size = 5) +
  xlab("Cutoff (cp)") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various complex. params")
ggsave("images/cv_dt.png", width = 8, height = 5)



#create model using optimal tree
model_tree <- rpart(PD~., cp = opt_cp, data =  heart_train, method = "class")

#plot the model - this really helps to understand how the algorithm works
rpart.plot(model_tree, type = 0) 

#generate predictions
preds_tree <- predict(model_tree, heart_test, type = "class") 

#confusion matrix
cm_tree <- confusionMatrix(preds_tree, heart_test$PD)

# save accuracy and sensitivity
accuracy_results <- rbind(accuracy_results,
                          data.frame(Method = "Decision Tree", 
                                     accuracy = cm_tree$overall["Accuracy"],
                                     sensitivity = cm_tree$byClass["Sensitivity"]))

###############
# random forest
###############

#will try a range of mtry values for the ideal forest
#10-fold CV will be used to obtain the optimal mtry

#Build model, use cv to get the ideal mtry
k <- 10
mtry = 1:11
accuracy_mtry <- matrix(nrow = k, ncol = length(mtry))

#create folds
set.seed(262, sample.kind = "Rounding")
ind <- createFolds(1:nrow(heart_train), k = k)

#perform cv
for (i in 1:k) {
  # create train and test sets for cv
  cv_train <- heart_train %>% slice(-ind[[i]])
  cv_test <- heart_train %>% slice(ind[[i]])
  
  # fill matrix with results (mean of accuracy and sensitivity) from decision tree 
  accuracy_mtry[i,] <- sapply(mtry, function(m){
    
    # create the decision tree
    set.seed(262, sample.kind = "Rounding")
    cv_mod_rf <- randomForest(PD ~.,
                              data = cv_train,
                              mtry = m)
    
    # obtain the predictions
    cv_preds_rf <- predict(cv_mod_rf, cv_test)
    
    # confusion matrix 
    cv_cm <- confusionMatrix(cv_preds_rf, cv_test$PD)
    
    # return the mean of the accuracy and sensitivity
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
  })
  # keep track of how many seeds have been run
  cat("fold",i,"out of 10 complete\n") 
}

#extract optimal mtry
opt_mtry <- median(mtry[which(min_rank(desc(colMeans(accuracy_mtry)))==1)])

#plot mtry vs accuracy
tibble(mtry = mtry, mean_accuracy = colMeans(accuracy_mtry)) %>%
  ggplot(aes(mtry, mean_accuracy)) +
  geom_line()+
  geom_point() +
  geom_point(aes(opt_mtry, max(mean_accuracy)), shape = 5, size = 5) +
  xlab("mtry") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for various mtry")
ggsave("images/cv_rf.png", width = 8, height = 5)


#build model using optimal mtry
model_rf <- randomForest(PD ~.,
                         data = heart_train,
                         mtry = opt_mtry)

#create predictions
preds_rf <- predict(model_rf, heart_test)

#confusion matrix
cm_rf <- confusionMatrix(preds_rf,heart_test$PD)

#save accuracy and sensitivity
accuracy_results <- rbind(accuracy_results,
                          data.frame(Method = "Random Forest",
                                     accuracy = cm_rf$overall["Accuracy"],
                                     sensitivity = cm_rf$byClass["Sensitivity"]))

#######################
# Support Vector Machine
#######################

#will try a range of cost values for the ideal model and store results in a matrix
#10-fold CV will be used to obtain the optimal cost


cost = seq(0.01, 1, 0.01)
accuracy_cost <- matrix(nrow = k, ncol = length(cost))

#create folds
set.seed(2014, sample.kind = "Rounding")
ind <- createFolds(1:nrow(heart_train), k = k)

#perform cv
for (i in 1:k) {
  # create train and test sets for cv
  cv_train <- heart_train %>% slice(-ind[[i]])
  cv_test <- heart_train %>% slice(ind[[i]])
  
  # fill matrix with results (mean of accuracy and sensitivity) from model
  accuracy_cost[i,] <- sapply(cost, function(c){
    
    # create the decision tree
    cv_mod_svm <- svm(PD ~.,
                      data = cv_train,
                      scale = T,
                      center = T,
                      kernel = "linear",
                      cost = c)
    
    # obtain the predictions 
    cv_preds_svm <- predict(cv_mod_svm, cv_test)
    
    # confusion matrix 
    cv_cm <- confusionMatrix(cv_preds_svm, cv_test$PD)
    
    # return the mean of the accuracy and sensitivity
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
  })
  # keep track of how many seeds have been run
  cat("fold",i,"out of 10 complete\n") 
}

#extract optimal cost
opt_cost <- median(cost[which(min_rank(desc(colMeans(accuracy_cost)))==1)])

#plot cost vs accuracy
tibble(cost = cost, mean_accuracy = colMeans(accuracy_cost)) %>%
  ggplot(aes(cost, mean_accuracy)) +
  geom_smooth()+
  geom_point() +
  geom_point(aes(opt_cost, max(mean_accuracy)), shape = 5, size = 5) +
  xlab("cost") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for various cost values")
ggsave("images/cv_svm.png", width = 8, height = 5)


#create model using optimal cost parameter
model_svm <- svm(PD ~.,
    data = train,
    scale = T,
    center = T,
    kernel = "linear",
    cost = opt_cost)

#generate predictions
preds_svm <- predict(model_svm, heart_test)

#confusion matrix
cm_svm <- confusionMatrix(preds_svm, heart_test$PD)

#store results
accuracy_results <- rbind(accuracy_results,
                          data.frame(Method = "Support Vector Machine",
                                     accuracy = cm_svm$overall["Accuracy"],
                                     sensitivity = cm_svm$byClass["Sensitivity"]))

##########
# ensemble
##########

#perform ensemble model by using the top 3 performing models
accuracy_results %>% mutate(mean_acc_sens = (accuracy + sensitivity)/2) %>% arrange(mean_acc_sens) # top 3 is Naive Bayes, Decision Tree and Random Forest

all_preds <- tibble(nb = preds_nb,
                    tree = preds_tree,
                    rf  = preds_rf)

# the predictions of the ensemble are obtained by majority votes
preds_ens <- apply(all_preds,1,function(x) names(which.max(table(x)))) %>%
  factor(levels = c("Yes","No"))

# confusion matrix (it actually performs worse than the random forest)
cm_ens <- confusionMatrix(preds_ens, heart_test$PD)

# save accuracy and sensitivity
accuracy_results <- rbind(accuracy_results,
                          data.frame(Method = "Ensemble",
                                     accuracy = cm_ens$overall["Accuracy"],
                                     sensitivity = cm_ens$byClass["Sensitivity"]))


# store all results (rmd)
results <- tibble(Method = c("Logistic Regression", "Naive Bayes", "Decision Tree","Random Forest","SVM","Ensemble"),
                  Accuracy = c(confusionMatrix(preds_glm_two, heart_test$PD)$overall["Accuracy"],
                               confusionMatrix(preds_nb, heart_test$PD)$overall["Accuracy"], 
                               confusionMatrix(preds_tree, heart_test$PD)$overall["Accuracy"],
                               confusionMatrix(preds_rf, heart_test$PD)$overall["Accuracy"],
                               confusionMatrix(preds_svm, heart_test$PD)$overall["Accuracy"],
                               confusionMatrix(preds_ens, heart_test$PD)$overall["Accuracy"]),
                  Sensitivity = c(confusionMatrix(preds_glm_two, heart_test$PD)$byClass["Sensitivity"], 
                                  confusionMatrix(preds_nb, heart_test$PD)$byClass["Sensitivity"],
                                  confusionMatrix(preds_tree, heart_test$PD)$byClass["Sensitivity"],
                                  confusionMatrix(preds_rf, heart_test$PD)$byClass["Sensitivity"],
                                  confusionMatrix(preds_svm, heart_test$PD)$byClass["Sensitivity"],
                                  confusionMatrix(preds_ens, heart_test$PD)$byClass["Sensitivity"])) %>%
  mutate(Mean = rowMeans(dplyr::select(., Accuracy, Sensitivity)))



accuracy_results %>%
  dplyr::select(accuracy, sensitivity) %>%
  mutate(means = rowMeans(.)) #Ensemble best model


#############################
# final model - ensemble
#############################

# the ensemble had the highest accuracy and sensitivity out of all of the models
# therefore, it is chosen to be the final model and is reconstructed using the diabetes data set
# we build the Naive Bayes, Decision Tree and Random Forest models with the validation set and ensemble these models for our final model

#############
# naive bayes
#############
set.seed(1, sample.kind = "Rounding")
model_nb_final <- train(PD~.,
                  method = 'nb',
                  data = train,
                  trControl = trainControl(method = 'cv', number = 10))

#create predictions
preds_nb_final <- predict(model_nb_final, validation)



##############
#decision tree
##############
k <- 10
cp = seq(0,0.5,0.01)
accuracy_cp_final <- matrix(nrow = k, ncol = length(cp))

#create folds
set.seed(2005, sample.kind = "Rounding")
ind <- createFolds(1:nrow(train), k = k)

for (i in 1:k) {
  # create train and test sets for cv
  cv_train <- train %>% slice(-ind[[i]])
  cv_test <- train %>% slice(ind[[i]])
  
  # fill matrix with results (mean of accuracy and sensitivity) from glm with cutoff p
  accuracy_cp_final[i,] <- sapply(cp, function(cp){
    
    # create the decision tree
    cv_mod_tree <- rpart(PD~., cp = cp, data =  cv_train, method = "class")
    
    # obtain the predictions
    cv_preds_tree <- predict(cv_mod_tree, cv_test, type = "class")
    
    # confusion matrix
    cv_cm <- confusionMatrix(cv_preds_tree, cv_test$PD)
    
    # return the mean of the accuracy and sensitivity
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
  })
  # keep track of how many seeds have been run
  cat("fold",i,"out of 10 complete\n") 
}

#extract optimal cp
opt_cp_final <- median(cp[which(min_rank(desc(colMeans(accuracy_cp_final)))==1)])

#plot cp vs accuracy
tibble(cp = cp, mean_accuracy = colMeans(accuracy_cp_final)) %>%
  ggplot(aes(cp, mean_accuracy)) +
  geom_smooth()+
  geom_point() +
  geom_point(aes(opt_cp, max(mean_accuracy)), shape = 5, size = 5) +
  xlab("Cutoff (cp)") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various complex. params")

#create model using optimal tree
model_tree_final <- rpart(PD~., cp = opt_cp, data =  train, method = "class")

#generate predictions
preds_tree_final <- predict(model_tree_final, validation, type = "class") 


###############
# Random Forest
###############

#Build model, use cv to get the ideal mtry
k <- 10
mtry = 1:11
accuracy_mtry_final <- matrix(nrow = k, ncol = length(mtry))

#create folds
set.seed(1, sample.kind = "Rounding")
ind <- createFolds(1:nrow(train), k = k)

#perform cv
for (i in 1:k) {
  # create train and test sets for cv
  cv_train <- train %>% slice(-ind[[i]])
  cv_test <- train %>% slice(ind[[i]])
  
  # fill matrix with results (mean of accuracy and sensitivity) from decision tree 
  accuracy_mtry_final[i,] <- sapply(mtry, function(m){
    
    # create the random forest
    set.seed(1, sample.kind = "Rounding")
    cv_mod_rf <- randomForest(PD ~.,
                              data = cv_train,
                              mtry = m)
    
    # obtain the predictions 
    cv_preds_rf <- predict(cv_mod_rf, cv_test)
    
    #confusion matrix
    cv_cm <- confusionMatrix(cv_preds_rf, cv_test$PD)
    
    # return the mean of the accuracy and sensitivity
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
  })
  # keep track of how many seeds have been run
  cat("fold",i,"out of 10 complete\n") 
}

#extract optimal mtry
opt_mtry_final <- median(mtry[which(min_rank(desc(colMeans(accuracy_mtry_final)))==1)])

model_rf_final <-randomForest(PD ~.,
                              data = train,
                              mtry = opt_mtry_final)

#create predictions
preds_rf_final <- predict(model_rf_final, validation)


################
# final ensemble
################
all_preds_final <- tibble(nb = preds_nb_final,
                    tree = preds_tree_final,
                    rf  = preds_rf_final)

# the predictions of the ensemble are obtained by majority votes
preds_ens_final <- apply(all_preds_final,1,function(x) names(which.max(table(x)))) %>%
  factor(levels = c("Yes","No"))

cm_ens_final <-confusionMatrix(preds_ens_final, validation$PD)

accuracy_results <- rbind(accuracy_results,
                          data.frame(Method = "Final Model: Ensemble (using validation set)", 
                                     accuracy = cm_ens_final$overall["Accuracy"],
                                     sensitivity = cm_ens_final$byClass["Sensitivity"]))

#print accuracy and sensitivity
confusionMatrix(preds_ens_final, validation$PD)$overall["Accuracy"]
confusionMatrix(preds_ens_final, validation$PD)$byClass["Sensitivity"]

