
## Practical Machine Learning
##   Course Project
## Guillermo Monge - Feb 2015

## Correct WLE Form Predictor


# get data
setwd('/Users/Will/Documents/Coursera/Data Science - JHU/Courses/8. Practical Machine Learning/Course Project')

data.date <- format(Sys.Date(), "%a %b %d %Y")

tr.link <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
ts.link <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(url=tr.link, destfile="data/pml-training.csv", method="curl")
download.file(url=tr.link, destfile="data/pml-testing.csv", method="curl")

# Load training data
tr.raw <- read.csv("data/pml-training.csv")

# Filter NA variables
col.noNA <- sapply(tr.raw, function(x) sum(is.na(x))) == 0
training <- tr.raw[col.noNA]
training <- training[,-(1:7)]

# Create control training (10 k-folds)
set.seed(9998)
tr.folds50 <- createFolds(training$classe, k=50, list=T)

# Fit models for 20 random folds
sample.folds <- sample(1:50, 20)
for (k in c(34,43,23,13)) {
    cat("\n===== Fold: ", k, " =====\n")
    
    # generate names
    fld.name <- paste("Fold", sprintf("%02d",k), sep='')
    tr.name <- paste('tr',sprintf("%02d",k),sep='')
    mod.name <- paste('cart',sprintf("%02d",k),sep='')
    VI.name <- paste('vars',sprintf("%02d",k),sep='')
    
    cat(" - generating fold...")
    tr <- training[tr.folds50[[fld.name]],]
    cat(' fold has ', dim(tr)[1], ' rows\n')
    assign(tr.name, tr)
    
    cat(" - fitting CART model on fold...\n")
    modFit <- train(classe~., data=tr, method='rpart')
    assign(mod.name, modFit)
    
    cat(" - getting variable importance...\n")
    v.imp <- varImp(modFit)
    assign(VI.name, rownames(v.imp$importance)[v.imp$importance > 0])
}

# get all variables with importance
all.vars <- c(vars09,vars12,vars33,vars42,vars35,vars03,vars17,vars38,vars29,vars12,
              vars34,vars43,vars23,vars13,vars10,vars44,vars21,vars05,vars21,vars18)
table(all.vars)

# Filter by variables with importance
imp.cols <- colnames(training) %in% all.vars
final.tr <- training[, imp.cols]
final.tr$classe <- training$classe

# Establish train control (10-folds) for final models
t.f <- createFolds(final.tr$classe, k=40, list=T)
tr.f01 <- final.tr[t.f$Fold01,]
dim(tr.f01)
    # 491 38

## Model Comparison
# Random Forest approach
modRF <- train(classe~., data=tr.f01, method='rf')
modRF2 <- train(classe~., data=tr.f09, method='rf')
# Naive Bayes approach
modNB <- train(classe~., data=tr.f01, method='nb')
modNB2 <- train(classe~., data=tr.f09, method='nb')
# CART approach
modCART <- train(classe~., data=tr.f01, method='rpart')
modCART2 <- train(classe~., data=tr.f09, method='rpart')

# Comparation function
compMatrix <- function(model, df) {
    preds <- predict(model, df)
    comp <- data.frame(actuals=df$classe, predicted=preds)
    table(comp)
}

# Comparison #1 (train1:-test:3)
compMatrix(modRF, tr.f03)
compMatrix(modNB, tr.f03)
compMatrix(modCART, tr.f03)

#Comparison #2 (train1:-test:9)
compMatrix(modRF, tr.f09)
compMatrix(modNB, tr.f09)
compMatrix(modCART, tr.f09)

#Comparison #3 (train5:-test:3)
compMatrix(modRF2, tr.f03)
compMatrix(modNB2, tr.f03)
compMatrix(modCART2, tr.f03)

#Comparison #4 (train5:-test:9)
compMatrix(modRF2, tr.f09)
compMatrix(modNB2, tr.f09)
compMatrix(modCART2, tr.f09)


## Final Model Fit
# Bigger RF
finalRF <- randomForest(formula=classe~., data=final.tr, ntree=500, do.trace=100)
finalRF


###Prediction
te.raw <- read.csv("data/pml-testing.csv")
predictions <- predict(finalRF, te.raw)
predictions

