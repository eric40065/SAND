generate_from_data = function(folder, sparsity = c(8, 12), noise_to_singal = 0.25, iidt = TRUE){
    # sparsity = c(8, 12); noise_to_singal = 0.25; iidt = TRUE
    
    set.seed(5170328)
    X_full_true  = as.matrix(data.table::fread(file = paste(folder, "/X_full_true.csv", sep = ""), header = FALSE))
    X_full_noise = as.matrix(data.table::fread(file = paste(folder, "/X_full_noise.csv", sep = ""), header = FALSE))
    T_full = as.matrix(data.table::fread(file = paste(folder, "/T_full.csv", sep = ""), header = FALSE))
    num_subjects = dim(X_full_true)[1]
    len_t = ncol(T_full)
    
    X_obs = T_obs = matrix(NA, num_subjects, sparsity[2])
    if(sparsity[1] == sparsity[2]){
        obs_per_sub = rep(sparsity[1], num_subjects)
    }else{
        obs_per_sub = sample(sparsity[1]:sparsity[2], num_subjects, replace = TRUE)
    }
    if(!iidt){center_list = sample(1:len_t, num_subjects, replace = T)}
    for(i in 1:num_subjects){
        if(iidt){
            index_now = sort(sample(1:len_t, obs_per_sub[i], replace = FALSE))
        }else{
            center = center_list[i]
            prob_list = dnorm((-len_t):len_t, mean = 0, sd = sd(1:len_t)/runif(1, 1, 2))
            prob = prob_list[(len_t - center + 1):(2 * len_t - center)]
            index_now = sort(sample(1:len_t, obs_per_sub[i], replace = FALSE, prob = prob/sum(prob)))
        }
        if(noise_to_singal > 0){
            X_obs[i, 1:obs_per_sub[i]] = X_full_noise[i, index_now]
        }else{
            X_obs[i, 1:obs_per_sub[i]] = X_full_true[i, index_now]
        }
        T_obs[i, 1:obs_per_sub[i]] = T_full[i, index_now]
    }
    X_obs = cbind(obs_per_sub, X_obs)
    
    folder = str_replace(folder, "IID", ifelse(iidt, "IID", "NonIID"))
    folder = ifelse(diff(sparsity) == 0, paste(folder, "/dense", sep = ""), paste(folder, "/sparse", sep = ""))
    folder = ifelse(noise_to_singal > 0, paste(folder, "/w_error", sep = ""), paste(folder, "/wo_error", sep = ""))
    data.table::fwrite(T_obs, file = paste(folder, "/T_obs.csv", sep = ""), row.names = FALSE, col.names = FALSE)
    data.table::fwrite(X_obs, file =  paste(folder, "/X_obs.csv", sep = ""), row.names = FALSE, col.names = FALSE)
    return(list(X_full_true = X_full_true, X_obs = X_obs, T_obs = T_obs, X_full_noise = X_full_noise))
}

