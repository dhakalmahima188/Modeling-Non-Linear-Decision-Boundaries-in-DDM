library(readr)

load("data/experiment-1-RDM.RData")

write.csv(new_object, file = "data/experiment-1-RDM.csv")

load("data/experiment-2-Flash.RData")

write.csv(saveit, file = "data/experiment-2-Flash.csv")

load("data/experiment-3-Flash.RData")

write.csv(saveit, file = "data/experiment-3-Flash.csv")
