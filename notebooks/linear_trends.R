library(funtimes) #Version 7.0
data_1 <- readRDS('df.Rds')
set.seed(8786)

name_list <- list()

for (x in colnames(data_1)){
  if (x=='Croatia - MIZ'){
    next
  }
  print(x)
  t <- c(data_1[[x]])
  pv <- notrend_test(t[!is.na(t)])$p.value
  name_list[[x]] <- pv
}

df <- as.data.frame(name_list)

saveRDS(df, file="df_R.Rds")
