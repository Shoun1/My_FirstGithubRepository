library(readr)
library(ggplot2)
df <- read_csv("ipl_dataset/Batting_PerMatchData_IPL_2023.csv")
kohli_df = subset(df,batsmanName == "Virat Kohli")
samson_df = subset(df,batsmanName == "Sanju Samson")
print(head(russell_df))
print(head(df$batsmanName))
#print(colnames(df))
#print(class(colnames(df)))
runs_vector <- df$runs[df$batsmanName == "Virat Kohli"]
runs_vector1 <- df$runs[df$batsmanName == "Sanju Samson"]
#runs_vector <- df$runs
print(length(runs_vector))
#print(typeof(df$SR))
#df$SR <- as.numeric(df$SR)
strikerate <- df$SR[df$batsmanName == "Virat Kohli"]
sr <- df$SR[df$batsmanName == "Sanju Samson"]
#strikerate <- df$SR
print(length(strikerate))
ob <- ggplot(data=kohli_df,aes(x = runs_vector,y = strikerate)) + geom_point()
ob1 <- ggplot(data=samson_df,aes(x = runs_vector1,y = sr)) + geom_point()
print(ob1)
print(ob)
#ob <- ggplot(data=df) + geom_point(mapping = aes(x = runs_vector,y = strikerate))
#print(ob)
rlang::last_trace()
