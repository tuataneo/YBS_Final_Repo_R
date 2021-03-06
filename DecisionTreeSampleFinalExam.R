#Kutuphaneleri indiriyoruz
# Yard�mc� Kutuphaneleri
install.packages("dplyr")
install.packages("ggplot2")
# Model Kutuphaneleri
install.packages("rpart")
install.packages("caret")
# Model Yorumlama/Gorsellestirme Kutuphaneleri
install.packages("rpart.plot")
install.packages("vip")
install.packages("pdp")
install.packages("rsample")

#Kutuphaneleri kullanmak i�in yukluyoruz
# Yard�mc� Kutuphaneleri
library(dplyr) # for data wrangling
library(ggplot2) # grafikleme i�in
# Model Kutuphaneleri
library(rpart) # karar a�ac� uygulamalar� i�in �al�san engine(Kutuphane)
library(caret) # karar a�ac� uygulamalar� i�in �al�san meta-engine
# Model Yorumlama/Gorsellestirme Kutuphaneleri
library(rpart.plot) # karar agaci i�im grafikleme
library(vip) # ozellik 
library(pdp) # ozellik efekleri i�in 
library(rsample)

set.seed(123)
ames <- AmesHousing::make_ames()
split <- initial_split(ames, prop = 0.7,
                       strata = "Sale_Price")
ames_train <- training(split)


ames_dt1 <- rpart(
  formula = Sale_Price ~ .,
  data = ames_train,
  method = "anova"
)
ames_dt1

rpart.plot(ames_dt1)
plotcp(ames_dt1)


ames_dt2 <- rpart(
  formula = Sale_Price ~ .,
  data = ames_train,
  method = "anova",
  control = list(cp = 0, xval = 10)
)

plotcp(ames_dt2)
abline(v = 11, lty = "dashed")



ames_dt1$cptable

ames_dt3 <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 20
)

ggplot(ames_dt3)

vip(ames_dt3, num_features = 40, bar = FALSE)

# K�smi bagimlilik grafikleri
p1 <- partial(ames_dt3, pred.var = "Gr_Liv_Area") %>% autoplot()
p2 <- partial(ames_dt3, pred.var = "Year_Built") %>% autoplot()
p3 <- partial(ames_dt3, pred.var = c("Gr_Liv_Area", "Year_Built")) %>%
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE,
              colorkey = TRUE, screen = list(z = -20, x = -60))
# Grafileri yan yana g�ster
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)


