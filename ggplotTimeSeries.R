# attempting ggplot stuff 

mod_df <- data.frame(
  year = dates,
  cases = AIDS_ts_mod
)


f1 <- data.frame(
  year = dates,
  cases = fcast1_AIDS$fitted
)


f2 <- data.frame(
  year = dates,
  cases = fcast2_AIDS$fitted
)


f3 <- data.frame(
  year = dates,
  cases = fcast3_AIDS$fitted
)



AIDS
ggplot(mod_df, aes(x = year, y=cases)) + 
  geom_line(data = mod_df, aes(x = year, y = cases), color = "black") +
  geom_line(data = fcast1_AIDS, aes(x = year, y = fitted), color = "blue") +
  xlab('year') +
  ylab('cases')



dates <- format(as.POSIXct(AIDS$Month.Reported.Code[1:112], tz="", format = "%Y/%M"), format="%M-%Y")