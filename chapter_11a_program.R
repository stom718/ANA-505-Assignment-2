# Hierarchical Bayes Part-Worth Estimation: Training and Test

# load market simulation utilities
##runs the market simulation program adn loads the file.
load(file="mtpa_market_simulation_utilities.RData")
## This function implemets an MCMC algorithm to estimate a hieraarchial multinomial logit model
## with a normal heterogenity distribution.
##can be installed using the install.packages()
library(ChoiceModelR)  # for Hierarchical Bayes Estimation
## caret is short for Classification and REgreesion Traning. This package contains functions to 
## streamline the model traning process for complex regression and classification problems.
library(caret)  # for confusion matrix function

# read in the data from a case study in computer choice.
##read the data from the csv file that contains all of the user responses from the survey questionnaire.
complete.data.frame <- read.csv("computer_choice_study.csv")

print.digits <- 2
# user-defined function for printing conjoint measures
## prints values rounded off to two decimal points or three by using an if function
if (print.digits == 2) 
  pretty.print <- function(x) {sprintf("%1.2f",round(x,digits = 2))} 
if (print.digits == 3) 
  pretty.print <- function(x) {sprintf("%1.3f",round(x,digits = 3))} 
 
# set up sum contrasts for effects coding
##Function 
options(contrasts=c("contr.sum","contr.poly"))

# employ a training-and-test regimen across survey sets/items 
## here you assign the test set ids to the following 4
test.set.ids <- c("3","7","11","15")  # select four sets/items
## the setdiff function compares the complte dta frame and 
training.set.ids <- setdiff(unique(complete.data.frame$setid),test.set.ids)
## using subset you specifiy where to extract the elements from  - here u set up the traning and test set
##data frame
training.data.frame <- 
  subset(complete.data.frame,subset=(setid %in% training.set.ids))
test.data.frame <- 
  subset(complete.data.frame,subset=(setid %in% test.set.ids))
## using the unique - you extart all unique elements with duplicates removed from the complete Data frame.
UniqueID <- unique(training.data.frame$id)
# set up zero priors
## The statement is assigning empty matrices with number of rows with unique values with number of columns as 13
cc.priors <- matrix(0,nrow=length(UniqueID),ncol=13) 

# we could use coefficients from aggregate model as starting values
# here we comment out the code needed to do that
# aggregate.cc.betas <- c(as.numeric(conjoint.results$coefficients)[2:7],
#  -sum(as.numeric(conjoint.results$coefficients)[2:7]),
#  as.numeric(conjoint.results$coefficients)[8:13])
# clone aggregate part-worths across the individuals in the study
# set up Bayesian priors
# cc.priors <- matrix(0,nrow=length(UniqueID),ncol=length(aggregate.cc.betas)) 
# for(index.for.ID in seq(along=UniqueID))
# cc.priors[index.for.ID,] <- aggregate.cc.betas

## assigning user defined column names to the matrices in cc.priors
colnames(cc.priors) <- c("A1B1","A1B2","A1B3","A1B4","A1B5","A1B6","A1B7",
  "A1B8","A2B1","A3B1","A4B1","A5B1","A6B1")

# note that the actual names are as follows: 
## AB names is a variable that has a list of actual names
AB.names <- c("Apple","Compaq","Dell","Gateway","HP","IBM","Sony","Sun",
  "Compatibility","Performance","Reliability","Learning","Price")

# set up run parameters for the MCMC
# using aggregate beta estimates to get started

truebetas <- cc.priors
cc.xcoding <- c(0,1,1,1,1,1)  # first variable categorical others continuous
cc.attlevels <- c(8,8,4,2,8,8) # test run with all attributes and levels
# no constraint for order on brand so 8x8 matrix of zeroes
## This section of the code assigns the signs of the attributes in the order of preference.Since brand is catergorical they are not assigned
## so signs are assigned.
c1 <- matrix(0,ncol=8,nrow=8)

# compatibility is ordered higher numbers are better
# continuous attributes have 1x1 matrix representation
c2 <- matrix(1, ncol = 1, nrow = 1, byrow = TRUE)

# performance is ordered higher numbers are better
# continuous attributes have 1x1 matrix representation
c3 <- matrix(1, ncol = 1, nrow = 1, byrow = TRUE)

# reliability is ordered higher numbers are better
# continuous attributes have 1x1 matrix representation
c4 <- matrix(1, ncol = 1, nrow = 1, byrow = TRUE)

# learning has expected order... higher learning times less valued
# continuous attributes have 1x1 matrix representation
c5 <- matrix(-1, ncol = 1, nrow = 1, byrow = TRUE)

# price has expected order... higher prices less valued
# continuous attributes have 1x1 matrix representation
c6 <- matrix(-1, ncol = 1, nrow = 1, byrow = TRUE)

cc.constraints <- list(c1,c2,c3,c4,c5,c6)

# controls for length of run and sampling from end of run
# cc.mcmc <- list(R = 10, use = 10) # fast trial run
# set run parameters 10000 total iterations with estimates based on last 2000
## here to reduce the time for the run R is set as 100 from 10,000 iterations and use to 20 from 2000.
cc.mcmc <- list(R = 100, use = 20) # run parameters

# run options
cc.options <- list(none=FALSE, save=TRUE, keep=1)

# set up the data frame for analysis
# redefine set ids so they are a complete set 1-12 as needed for HB functions
## here each setid is replaced with 3, 7 & 11
training.data.frame$newsetid <- training.data.frame$setid
training.data.frame$newsetid <- ifelse((training.data.frame$newsetid == 16),
  3,training.data.frame$newsetid)
training.data.frame$newsetid <- ifelse((training.data.frame$newsetid == 14),
  7,training.data.frame$newsetid)
training.data.frame$newsetid <- ifelse((training.data.frame$newsetid == 13),
  11,training.data.frame$newsetid)

UnitID <- training.data.frame$id
Set <- as.integer(training.data.frame$newsetid)
Alt <- as.integer(as.factor(training.data.frame$position))
X_1 <- as.integer(as.factor(training.data.frame$brand)) # categories by brand
X_2 <- as.integer(training.data.frame$compat)  # integer values 1 to 8
X_3 <- as.integer(training.data.frame$perform)  # integer values 1 to 4
X_4 <- as.integer(training.data.frame$reliab)  # integer values 1 to 2
X_5 <- as.integer(training.data.frame$learn)  # integer values 1 to 8
X_6 <- as.integer(training.data.frame$price)  # integer values 1 to 8
y <- as.numeric(training.data.frame$choice)  # using special response coding
## This section of the code converting all the catergorical variables into intergers.
## Creating a copy of the traning data frame into cc.data
cc.data <- data.frame(UnitID,Set,Alt,X_1,X_2,X_3,X_4,X_5,X_6,y)

# now for the estimation... be patient
set.seed(9999)  # for reproducible results
## 
## This section of the code runs the choicemodel r package and stores the beta values as individual part worths in the file RBetas.csv
## The choice model R runs an mcmc alogorithm which esitmates the hierarchial bayes multinomoial logit model
out <- choicemodelr(data=cc.data, xcoding = cc.xcoding, 
  mcmc = cc.mcmc, options = cc.options, constraints = cc.constraints)

# out provides a list for the posterior parameter estimates 
# for the runs sampled (use = 2000)

# the MCMC beta parameter estimates are traced on the screen as it runs

# individual part-worth estimates are provided in the output file RBetas.csv
# the final estimates are printed to RBetas.csv with columns labeled as
#  A1B1 = first attribute first level
#  A1B2 = first attribute second level
#  ....
#  A2B1 = second attribute first level
#  ....
# gather data from HB posterior parameter distributions
# we imposed constraints on all continuous parameters so we use betadraw.c
posterior.mean <- matrix(0, nrow = dim(out$betadraw.c)[1], 
  ncol = dim(out$betadraw.c)[2])
posterior.sd <- matrix(0, nrow = dim(out$betadraw.c)[1], 
  ncol = dim(out$betadraw.c)[2])
## This section of the code is iterating over each column and row
for(index.row in 1:dim(out$betadraw.c)[1])
for(index.col in 1:dim(out$betadraw.c)[2]) { 
  posterior.mean[index.row,index.col] <- 
    mean(out$betadraw.c[index.row,index.col,])
  posterior.sd[index.row,index.col] <- 
    sd(out$betadraw.c[index.row,index.col,])
  }

# HB program uses effects coding for categorical variables and
# mean-centers continuous variables across the levels appearing in the data
# working with data for one respondent at a time we compute predicted choices
# for both the training and test choice sets

## creates a design matrix 
create.design.matrix <- function(input.data.frame.row) {
  xdesign.row <- numeric(12)
  if (input.data.frame.row$brand == "Apple") 
    xdesign.row[1:7] <- c(1,0,0,0,0,0,0)  
  if (input.data.frame.row$brand == "Compaq") 
    xdesign.row[1:7] <- c(0,1,0,0,0,0,0)  
  if (input.data.frame.row$brand == "Dell") 
    xdesign.row[1:7] <- c(0,0,1,0,0,0,0)  
  if (input.data.frame.row$brand == "Gateway") 
    xdesign.row[1:7] <- c(0,0,0,1,0,0,0)  
  if (input.data.frame.row$brand == "HP") 
    xdesign.row[1:7] <- c(0,0,0,0,1,0,0)  
  if (input.data.frame.row$brand == "IBM") 
    xdesign.row[1:7] <- c(0,0,0,0,0,1,0)  
  if (input.data.frame.row$brand == "Sony") 
    xdesign.row[1:7] <- c(0,0,0,0,0,0,1)  
  if (input.data.frame.row$brand == "Sun") 
    xdesign.row[1:7] <- c(-1,-1,-1,-1,-1,-1,-1)    
  
  xdesign.row[8] <- input.data.frame.row$compat -4.5 
  xdesign.row[9] <- input.data.frame.row$perform -2.5
  xdesign.row[10] <- input.data.frame.row$reliab -1.5 
  xdesign.row[11] <- input.data.frame.row$learn -4.5
  xdesign.row[12] <- input.data.frame.row$price -4.5 
  t(as.matrix(xdesign.row))  # return row of design matrix
  }

# evaluate performance in the training set
training.choice.utility <- NULL  # initialize utility vector
# work with one row of respondent training data frame at a time
# create choice prediction using the individual part-worths
## This section of the code tests the utility of the prediction model. 
## The percentage of chocies correctly predicted is the sensitivity of the model
## Iterating over each and every index of the id and posterior means are assigned to variable this.id.part.worths
list.of.ids <- unique(training.data.frame$id)
for (index.for.id in seq(along=list.of.ids)) {
  this.id.part.worths <- posterior.mean[index.for.id,] 
  ## Creating a subset of the traning data frame to match with the list of Id's
  this.id.data.frame <- subset(training.data.frame, 
    subset=(id == list.of.ids[index.for.id]))
  for (index.for.profile in 1:nrow(this.id.data.frame)) {   
    training.choice.utility <- c(training.choice.utility,
      create.design.matrix(this.id.data.frame[index.for.profile,]) %*%
      this.id.part.worths)
    }  
  }  
## Based on the values of training.choice.utility and assinging yes or no lables to training.actual.choice
training.predicted.choice <- 
  choice.set.predictor(training.choice.utility)
training.actual.choice <- factor(training.data.frame$choice, levels = c(0,1), 
  labels = c("NO","YES"))  
# look for sensitivity > 0.25 for four-profile choice sets 
training.set.performance <- confusionMatrix(data = training.predicted.choice, 
  reference = training.actual.choice, positive = "YES")
# report choice prediction sensitivity for training data
cat("\n\nTraining choice set sensitivity = ",
  sprintf("%1.1f",training.set.performance$byClass[1]*100)," Percent",sep="")

# evaluate performance in the test set
test.choice.utility <- NULL  # initialize utility vector
# work with one row of respondent test data frame at a time
# create choice prediction using the individual part-worths
list.of.ids <- unique(test.data.frame$id)
for (index.for.id in seq(along=list.of.ids)) {
  this.id.part.worths <- posterior.mean[index.for.id,] 
  this.id.data.frame <- subset(test.data.frame, 
    subset=(id == list.of.ids[index.for.id]))
  for (index.for.profile in 1:nrow(this.id.data.frame)) {    
    test.choice.utility <- c(test.choice.utility,
      create.design.matrix(this.id.data.frame[index.for.profile,]) %*%
      this.id.part.worths)
    }  
  }  

test.predicted.choice <- 
  choice.set.predictor(test.choice.utility)
test.actual.choice <- factor(test.data.frame$choice, levels = c(0,1), 
  labels = c("NO","YES"))  
# look for sensitivity > 0.25 for four-profile choice sets  
test.set.performance <- confusionMatrix(data = test.predicted.choice, 
  reference = test.actual.choice, positive = "YES")
# report choice prediction sensitivity for test data
cat("\n\nTest choice set sensitivity = ",
  sprintf("%1.1f",test.set.performance$byClass[1]*100)," Percent",sep="")

# suggestions for students
# having demonstrated the predictive power of the HB model...
# return to the complete set of 16 choice sets to obtain 
# part-worths for individuals based upon the complete survey
# (the next program will provide guidance on how to do this)
# after estimating part-worths for individuals, average across
# individuals to obtain an aggregate profile of conjoint measures
# standardize the aggregate part-worths and display them
# on a spine chart using the spine chart plotting utility
# provided in the appendix of code and utilities
# interpret the spine chart, compare attribute importance values
# compare the brands, compute brand equity for each brand
# relative to each of the other brands in the study
