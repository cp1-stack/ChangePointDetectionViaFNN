# install.packages("cpm") 
library(cpm)
cps_data <- read.csv("ChangePoints_data_test_betas.csv", header=F)
cps_data$V1
plot(c(1:1000),cps_data$V1)
lines(c(1:1000),cps_data$V1)

cpm.res = processStream(cps_data$V1, cpmType = "Student")

plot(cps_data$V1, type = "l", col = "steelblue", lwd = 2)
abline(v = cpm.res$changePoints, lwd = 3.5, col = "red")
print(cpm.res$changePoints)
###############################################
library(InspectChangepoint)
cps_data <- read.csv("ChangePoints_data_test_betas.csv", header=F)
cusum.transform(cps_data$V1)
ret <- inspect(cps_data$V1)
ret
summary(ret)
plot(ret)
# plot(c(1:500),cps_data$V1)