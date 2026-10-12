# HR Analytics Prediction

Month <- c(1, 2, 3, 4, 5)
Employees <- c(100, 110, 120, 130, 140)

data <- data.frame(Month, Employees)

model <- lm(Employees ~ Month, data = data)

newdata <- data.frame(Month = 6)

prediction <- predict(model, newdata)

print(prediction)
