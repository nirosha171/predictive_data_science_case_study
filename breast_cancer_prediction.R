# Breast Cancer Prediction using Logistic Regression

Age <- c(25, 35, 45, 55, 65)
TumorSize <- c(2, 3, 4, 5, 6)
Cancer <- c(0, 0, 1, 1, 1)

data <- data.frame(Age, TumorSize, Cancer)

model <- glm(Cancer ~ Age + TumorSize,
             data = data,
             family = binomial)

newdata <- data.frame(Age = 40, TumorSize = 3)

prediction <- predict(model, newdata, type = "response")

print(prediction)
