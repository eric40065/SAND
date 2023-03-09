rm(list = ls()); gc();
library(data.table)
substrRight = function(x, n){substr(x, nchar(x) - n + 1, nchar(x))}
get_time_in_hour = function(x){
    # x = block_now$tstp
    x = as.character(x)
    year = as.integer(substr(x, 1, 4))
    month = as.integer(substr(x, 6, 7))
    day = as.integer(substr(x, 9, 10))
    hour = as.integer(substr(x, 12, 13))
    minute = as.integer(substr(x, 15, 16))
    
    day_per_month = cumsum(c(0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30))
    result = 24 * ((year == 2012) * (month > 2) + (year > 2012)) + (year - 2011) * 365 * 24 + day_per_month[month] * 24 + (day - 1) * 24 + hour + minute/60
    return(result)
}

i = 0
num_subjects = 0
block = NULL
for(i in 0:111){
    print(i)
    block_now = fread(paste("/Users/eric/Desktop/UCD/TransFD/real_data/halfhourly_dataset/block_", i, ".csv", sep = ""), colClasses = rep("character", 3), sep = ",")
    block_now = block_now[block_now$`energy(kWh/hh)` != "Null", ]
    block_now$`energy(kWh/hh)` = as.numeric(block_now$`energy(kWh/hh)`)
    
    tmp = as.vector(table(as.integer(substrRight(block_now$LCLid, 6))))
    block_now$LCLid = rep(1:length(tmp), as.vector(tmp)) + num_subjects
    block_now$tstp = get_time_in_hour(block_now$tstp)
    num_subjects = block_now$LCLid[length(block_now$LCLid)]

    block = rbind(block, block_now)
    gc()
}
fwrite(block, file = "/Users/eric/Desktop/UCD/TransFD/real_data/raw.csv")
block$tstp = block$tstp - min(block$tstp) + 9 # after this line, the first data is at 11/23/2011 00:00
fwrite(block, file = "/Users/eric/Desktop/UCD/TransFD/real_data/data.csv")
