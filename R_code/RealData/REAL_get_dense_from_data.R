rm(list = ls()); gc();
library(data.table)
data = fread("/Users/eric/Desktop/UCD/TransFD/Data/Raw/RealData/data.csv", sep = ",")
colnames(data) = c("id", "t", "x")
split_data = split(data, data$id)

num_tpts = diff(c(0, which(diff(data$id) == 1), nrow(data)))
num_subject = length(num_tpts)
cum_num_tpts = c(0, cumsum(num_tpts))
time_range = cbind(data$t[cum_num_tpts + 1][-length(cum_num_tpts)], data$t[cum_num_tpts])

time_grid = rep(0, max(time_range) * 2)
for(i in 1:num_subject){
    index_now = (2*time_range[i, 1]) : (2*time_range[i, 2])
    time_grid[index_now] = time_grid[index_now] + 1
}

plot(time_grid, type = "l")
time_shared = c(min((1:max(time_range) * 2)[time_grid == max(time_grid)]), max((1:max(time_range) * 2)[time_grid == max(time_grid)]))
time_shared = c(17304, 17399)
time_shared = (time_shared[1]:time_shared[2])/2

rough_data = lapply(split_data, function(small_data){
    return(small_data[small_data$t %in% time_shared, ])
})

final_tpts = length(time_shared)
full_observed = which(as.vector(sapply(rough_data, function(x){dim(x)[1]})) == final_tpts)
x = matrix(0, length(full_observed), final_tpts); counter = 1
for(i in full_observed){
    x[counter, ] = rough_data[[i]]$x
    counter = counter + 1
}

t = matrix(0:(final_tpts - 1), length(full_observed), final_tpts, byrow = TRUE)/(final_tpts - 1)
set.seed(11242022)
permutation = sample(1:dim(x)[1], dim(x)[1], replace = FALSE)

data.table::fwrite(x[permutation, ], file = "/Users/eric/Desktop/UCD/TransFD/Data/IID/RealData/UK/X_full_noise.csv", row.names = FALSE, col.names = FALSE)
data.table::fwrite(t[permutation, ], file = "/Users/eric/Desktop/UCD/TransFD/Data/IID/RealData/UK/T_full.csv", row.names = FALSE, col.names = FALSE)
