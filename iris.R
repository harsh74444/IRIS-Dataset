data(iris)
library(caret)
index <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
testset <- iris[-index,]
trainset <- iris[index,]

# Performing EDA (Exploratory Data Analysis)

dim(trainset)
str(trainset)
summary(trainset)
levels(trainset$Species)

hist(trainset$Sepal.Length)
ggplot(data = trainset, aes(Sepal.Width, Sepal.Length)) + geom_point()

# Boxplot
par(mfrow = c(1,4))
  for (i in 1:4) {
  boxplot(trainset[,i], main = names(trainset)[i])
}

par(mfrow = c(2,2))
  for (i in 1:4){
    boxplot(trainset[,i], main = names(trainset)[i])
  }

g <- ggplot(data = trainset, aes(Sepal.Length, Sepal.Width))+
  geom_point(aes(color = Species, shape = Species)) + geom_smooth(method = "lm")
g


gg <- ggplot(data = trainset, aes(x = Species, y = Sepal.Length)) +
  geom_boxplot(aes(fill = Species)) + 
  stat_summary(fun.y = mean, shape = 5, size = 4, geom = 'point') +
  ggtitle("IRIS BOXPLOT")
gg

library(ggthemes)
histogram <- ggplot(data = iris, aes(Sepal.Width)) +
  geom_histogram(aes(fill = Species), color = 'black', binwidth = 0.1) +
  theme_economist() + xlab("Sepal Width") + ylab("Frequency")

print(histogram)

#Faceting: Creating multiple charts in one plot

facet <- ggplot(data = iris, aes(Sepal.Length, Sepal.Width, color = Species)) + 
  geom_point(aes(shape = Species), size = 1.5) +
  geom_smooth(method = 'lm') + theme_fivethirtyeight() +
  facet_grid(.~Species) #Along rows
facet

print(model.rpart)


# Growing a tree
library(rpart)
control = rpart.control(minsplit = 20, cp = 0.01)
tree <- rpart(Species ~ Sepal.Length + Sepal.Width + Petal.Length 
              + Petal.Width, data = trainset, method = "class", control = control)

plot(tree, uniform = TRUE)
text(tree, use.n = TRUE, cex = 0.8)


##------------------------------- RANDOM FOREST --------------------------------##

#Random Forest
library(randomForest)
forest_1 <- randomForest(Species ~., data = trainset, importance = TRUE)
forest_1
forest_2 <- randomForest(Species ~., data = trainset, mtry = 6, ntree = 400,
                           importance = TRUE)

forest_2
predict_train <- predict(forest_2, trainset, type = "class")
table(predict_train, trainset$Species)
predict_test <- predict(forest_2, testset, type = "class")
table(predict_test, testset$Species)
mean(predict_test == testset$Species)
importance(forest_2)
varImpPlot(forest_2)
predict_2 <- predict(forest_2, testing, type = "class")
predict_2
library(rpart)
library(caret)
confusionMatrix(predict_test, testset$Species)


##---------------------------- END OF RANDOM FOREST --------------------------##

##--------------------------- KNN ---------------------------------------##

library(caret)
set.seed(7)
model_knn <- train(Species~., data = trainset, method = "knn",metric=metric, 
                   trcontrol = control)
predict_knn <- predict(model_knn, data = testset)
confusionMatrix(predict_knn, testset$Species)
