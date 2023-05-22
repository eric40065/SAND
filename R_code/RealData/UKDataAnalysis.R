library(plotly); library(fdapace); library(mice); library(stringr); library(data.table); 
#setwd("/Users/eric/Desktop/UCD/TransFD")
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
UK_data_appregation = function(){
    block = NULL
    for(i in 0:111){
        block_now = fread(paste("./Data/UK_Raw/halfhourly_dataset/block_", i, ".csv", sep = ""), colClasses = rep("character", 3), sep = ",")
        block_now = block_now[block_now$`energy(kWh/hh)` != "Null", ]
        block_now$`energy(kWh/hh)` = as.numeric(block_now$`energy(kWh/hh)`)
        
        tmp = as.vector(table(as.integer(substrRight(block_now$LCLid, 6))))
        block_now$LCLid = rep(1:length(tmp), as.vector(tmp)) + num_subjects
        block_now$tstp = get_time_in_hour(block_now$tstp)
        num_subjects = block_now$LCLid[length(block_now$LCLid)]
        
        block = rbind(block, block_now)
        gc()
    }
    # fwrite(block, file = "./raw.csv")
    block$tstp = block$tstp - min(block$tstp) + 9 # after this line, the first data is at 11/23/2011 00:00
    fwrite(block, file = "./Data/UK_raw/data.csv")
    return(NULL)
}
UK_get_dense_from_raw = function(){
    data = fread("./Data/UK_Raw/data.csv", sep = ",")
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
    
    # time_shared = c(min((1:max(time_range) * 2)[time_grid == max(time_grid)]), max((1:max(time_range) * 2)[time_grid == max(time_grid)]))
    time_shared = c(17304, 17399) # slice out data between 11/13 and 11/14
    time_shared = (time_shared[1]:time_shared[2])/2
    
    rough_data = lapply(split_data, function(small_data){
        return(small_data[small_data$t %in% time_shared, ])
    })
    
    final_tpts = length(time_shared)
    full_observed = which(as.vector(sapply(rough_data, function(x){dim(x)[1]})) == final_tpts)
    x = matrix(0, length(full_observed), final_tpts, byrow = TRUE); counter = 1
    for(i in full_observed){
        x[counter, ] = rough_data[[i]]$x
        counter = counter + 1
    }
    
    t = matrix(0:(final_tpts - 1), length(full_observed), final_tpts, byrow = TRUE)/(final_tpts - 1)
    set.seed(11242022)
    permutation = sample(1:dim(x)[1], dim(x)[1], replace = FALSE)
    data.table::fwrite(x[permutation, ], file = "./Data/IID/RealData/UK/X_full_noise.csv", row.names = FALSE, col.names = FALSE)
    data.table::fwrite(t[permutation, ], file = "./Data/IID/RealData/UK/T_full.csv", row.names = FALSE, col.names = FALSE)
    return(NULL)
}
get_sample_from_dense = function(folder, sparsity = c(8, 12), iidt = TRUE){
    set.seed(5170328)
    X_full_noise = as.matrix(data.table::fread(file = paste(folder, "/X_full_noise.csv", sep = ""), header = FALSE))
    T_full = as.matrix(data.table::fread(file = paste(folder, "/T_full.csv", sep = ""), header = FALSE))
    num_subjects = dim(X_full_noise)[1]
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
        X_obs[i, 1:obs_per_sub[i]] = X_full_noise[i, index_now]
        T_obs[i, 1:obs_per_sub[i]] = T_full[i, index_now]
    }
    X_obs = cbind(obs_per_sub, X_obs)
    
    folder = str_replace(folder, "IID", ifelse(iidt, "IID", "NonIID"))
    folder = ifelse(diff(sparsity) == 0, paste(folder, "/dense", sep = ""), paste(folder, "/sparse", sep = ""))
    data.table::fwrite(T_obs, file = paste(folder, "/T_obs.csv", sep = ""), row.names = FALSE, col.names = FALSE)
    data.table::fwrite(X_obs, file =  paste(folder, "/X_obs.csv", sep = ""), row.names = FALSE, col.names = FALSE)
    return(list(X_obs = X_obs, T_obs = T_obs, X_full_noise = X_full_noise))
}
get_design_plot = function(time_list){
    time_grid = unique(sort(unlist(time_list)))
    time_len = length(time_grid)
    result = matrix(0, time_len, time_len)
    for(i in 1:length(time_list)){
        now_time = round(time_list[[i]] * (time_len - 1)) + 1
        result[now_time, now_time] = result[now_time, now_time] + 1
    }
    diag_result = diag(result)
    diag(result)[1] = mean(c(result[1, 2], result[2, 1]))
    for(i in 2:time_len){diag(result)[i] = mean(c(result[i - 1, i], result[i, i - 1]))}
    par(mar=c(5.1, 4.1, 4.1, 4.1)) # adapt margins
    return(list(diag_result, result))
}
myPACE = function(do_PACE, X_obs, T_obs, X_full_true, X_full_noise, index, optns = NULL, Transformer = FALSE){
    if(do_PACE){
        num_subjects = length(X_obs)
        PACE_fit = FPCA(X_obs[index$training], T_obs[index$training], optns = optns)
        PACE_est = predict(PACE_fit, X_obs, T_obs)[[2]]
        if(Transformer){index$training = setdiff(index$training, index$testing)}
        MSE_train = mean((PACE_est[index$training, ] - X_full_true[index$training, ])^2)
        MSE_test = mean((PACE_est[index$testing, ] - X_full_true[index$testing, ])^2)
        MSE_train = c(MSE_train, mean((PACE_est[index$training, ] - X_full_noise[index$training, ])^2))
        MSE_test = c(MSE_test, mean((PACE_est[index$testing, ] - X_full_noise[index$testing, ])^2))
        return(list(MSE_train = MSE_train, MSE_test = MSE_test, EST_train = PACE_est[index$training, ], EST_test = PACE_est[index$testing, ], fit = PACE_fit))
    }else{
        return(list(MSE_train = NA, MSE_test = NA, EST_train = NA, EST_test = NA))
    }
}
my1DS = function(do_1DS, X_obs, T_obs, X_full_true, X_full_noise, index, obs_per_sub){
    if(do_1DS){
        n_train = length(index$training)
        subsamples = ceiling((runif(n_train)/5) * obs_per_sub[1:n_train])
        cv_pts_list = lapply(1:n_train, FUN = function(i){sort(sample(1:obs_per_sub[i], subsamples[i]))})
        not_cv_pts_list = lapply(1:n_train, FUN = function(i){setdiff(1:obs_per_sub[i], cv_pts_list[[i]])})
        
        bw_list = exp(pracma::linspace(log(1e-3/mean(obs_per_sub)), log(10/mean(obs_per_sub)), 30))
        CV_list = bw_list * 0
        counter_1D = 1
        for(bw in bw_list){
            error_now = 0
            for(i in 1:n_train){
                cv_pts_now = cv_pts_list[[i]]
                not_cv_pts_now = not_cv_pts_list[[i]]
                Xout = Lwls1D(bw, xin = T_obs[[i]][not_cv_pts_now], yin = X_obs[[i]][not_cv_pts_now], xout = T_obs[[i]][cv_pts_now], kernel_type = "gauss")
                error_now = error_now + sum((Xout - X_obs[[i]][cv_pts_now])^2)
            }
            CV_list[counter_1D] = error_now
            counter_1D = counter_1D + 1
        }
        bw = bw_list[which.min(CV_list)]
        IDS_est = matrix(0, length(X_obs), dim(X_full_true)[2])
        timegird = unique(sort(unlist(T_obs)))
        for(i in 1:dim(IDS_est)[1]){
            Xout = Lwls1D(bw, xin = T_obs[[i]], yin = X_obs[[i]], xout = timegird, kernel_type = "gauss")
            IDS_est[i, ] = Xout
        }
        MSE_train = mean((IDS_est[index$training, ] - X_full_true[index$training, ])^2)
        MSE_test = mean((IDS_est[index$testing, ] - X_full_true[index$testing, ])^2)
        MSE_train = c(MSE_train, mean((IDS_est[index$training, ] - X_full_noise[index$training, ])^2))
        MSE_test = c(MSE_test, mean((IDS_est[index$testing, ] - X_full_noise[index$testing, ])^2))
        return(list(MSE_train = MSE_train, MSE_test = MSE_test, EST_train = IDS_est[index$training, ], EST_test = IDS_est[index$testing, ]))
    }else{
        return(list(MSE_train = NA, MSE_test = NA, EST_train = NA, EST_test = NA))
    }
}
myMICE = function(do_MICE, X_obs, T_obs, X_full_true, X_full_noise, index, fast){
    if(do_MICE){
        timegird = unique(sort(unlist(T_obs)))
        X_MICE = matrix(NA, length(X_obs), length(timegird))
        for(i in 1:length(X_obs)){X_MICE[i, round(T_obs[[i]] * (length(timegird) - 1) + 1)] = X_obs[[i]]}
        if(fast){m = 1; maxit = 1}
        if(!fast){m = floor(sum(is.na(X_MICE))/prod(dim(X_MICE)) * 100 / 9) * 9; maxit = 5}
        mice.fit = futuremice(X_MICE, m = m, maxit = maxit, method = "pmm", n.core = 9, cl.type = "FORK", n.imp.core = m/9, diagnostics = FALSE, remove_collinear = FALSE)
        suppressMessages(MICE_est <- as.matrix(complete(mice.fit, "repeated")))
        MICE_est = sapply(1:length(timegird), function(j){rowSums(MICE_est[, ((j - 1)*m+1):(j*m)])}) / m
        MSE_train = mean((MICE_est[index$training, ] - X_full_true[index$training, ])^2, na.rm = TRUE)
        MSE_test = mean((MICE_est[index$testing, ] - X_full_true[index$testing, ])^2, na.rm = TRUE)
        MSE_train = c(MSE_train, mean((MICE_est[index$training, ] - X_full_noise[index$training, ])^2, na.rm = TRUE))
        MSE_test = c(MSE_test, mean((MICE_est[index$testing, ] - X_full_noise[index$testing, ])^2, na.rm = TRUE))
        return(list(MSE_train = MSE_train, MSE_test = MSE_test, EST_train = MICE_est[index$training, ], EST_test = MICE_est[index$testing, ]))
    }else{
        return(list(MSE_train = NA, MSE_test = NA, EST_train = NA, EST_test = NA))
    }
}
do_analysis = function(data_name_list = c("UK"), iidt_list = c("IID", "NonIID"), dense_sparse_list = c("dense", "sparse"),
                       split = c(90, 5, 5), do_PACE = TRUE, do_1DS = TRUE, do_MICE = TRUE, save_plot = FALSE, fast = FALSE){
    
    # data_name_list = c("UK"); iidt_list = c("IID", "NonIID"); dense_sparse_list = c("dense", "sparse");
    # split = c(90, 5, 5); do_PACE = TRUE; do_1DS = TRUE; do_MICE = TRUE; save_plot = FALSE; fast = FALSE;
    
    fit_object_3 = list(fitPACE = NA, fit1DS = NA, fitMICE = NA)
    fit_object_2 = setNames(rep(list(fit_object_3), length(dense_sparse_list)), dense_sparse_list)
    fit_object_1 = setNames(rep(list(fit_object_2), length(data_name_list)), data_name_list)
    fit_object = setNames(rep(list(fit_object_1), length(iidt_list)), iidt_list)
    
    fit_object_1 = setNames(rep(list(list(MSE_train = NA, MSE_test = NA)), length(data_name_list)), data_name_list)
    MSEs = setNames(rep(list(fit_object_1), length(iidt_list)), iidt_list)
    # iidt = "IID"; data_name = "UK"; dense_sparse = dense_sparse_list[1]
    true_folder = "./Data/IID/RealData"
    for(iidt in iidt_list){
        obs_folder  = paste("./Data", iidt, "RealData", sep = "/")
        for(data_name in data_name_list){
            true_datafolder = paste(true_folder, data_name, sep = "/")
            obs_datafolder  = paste(obs_folder,  data_name, sep = "/")
            
            # get data in full grid
            X_full_noise = as.matrix(data.table::fread(file = paste(true_datafolder, "/X_full_noise.csv", sep = ""), header = FALSE))
            
            ## auxiliary
            # MSE
            MSE_train = matrix(0, 3, 2)
            rownames(MSE_train) = c("Pace", "1DS", "Mice")
            colnames(MSE_train) = c("Dense", "Sparse")
            MSE_test = MSE_train
            counter_col = 0
            
            for(dense_sparse in dense_sparse_list){# aaa}}}
                print(iidt)
                print(data_name)
                print(dense_sparse)
                counter_col = counter_col + 1
                
                # get data
                obs_datafolder_de = paste(obs_datafolder, dense_sparse, sep = "/")
                X_obs = as.matrix(data.table::fread(file = paste(obs_datafolder_de, "/X_obs.csv", sep = ""), header = FALSE))
                T_obs = as.matrix(data.table::fread(file = paste(obs_datafolder_de, "/T_obs.csv", sep = ""), header = FALSE))
                
                # preprocessing
                num_subjects = dim(X_obs)[1]; n_train = round(num_subjects * sum(split[1:2])/sum(split))
                index = list(training = 1:n_train, testing = (n_train + 1):num_subjects)
                obs_per_sub = X_obs[, 1]
                X_obs = X_obs[, -1]
                X_obs = lapply(1:num_subjects, function(i){return(X_obs[i, 1:obs_per_sub[i]])})
                T_obs = lapply(1:num_subjects, function(i){return(T_obs[i, 1:obs_per_sub[i]])})
                
                # PACE
                optns = list(dataType = "Sparse", usergrid = TRUE, maxK = 10, methodBwCov = "GCV", plot = TRUE)
                if(fast){optns = list(dataType = "Sparse", usergrid = TRUE, maxK = 1, userBwCov = 0.03, plot = TRUE)}
                fitPACE = myPACE(do_PACE, X_obs, T_obs, X_full_noise, X_full_noise, index, optns)
                MSE_train      [1, counter_col] = fitPACE$MSE_train[1]
                MSE_test       [1, counter_col] = fitPACE$MSE_test[1]
                fit_object[[iidt]][[data_name]][[counter_col]]$fitPACE = fitPACE
                
                # 1D smoother
                fit1DS = my1DS(do_1DS, X_obs, T_obs, X_full_noise, X_full_noise, index, obs_per_sub)
                MSE_train      [2, counter_col] = fit1DS$MSE_train[1]
                MSE_test       [2, counter_col] = fit1DS$MSE_test[1]
                fit_object[[iidt]][[data_name]][[counter_col]]$fit1DS = fit1DS
                
                # MICE
                fitMICE = myMICE(do_MICE, X_obs, T_obs, X_full_noise, X_full_noise, index, fast)
                MSE_train      [3, counter_col] = fitMICE$MSE_train[1]
                MSE_test       [3, counter_col] = fitMICE$MSE_test[1]
                fit_object[[iidt]][[data_name]][[counter_col]]$fitMICE = fitMICE
            }
            MSEs[[iidt]][[data_name]]$MSE_train = MSE_train * 1e3
            MSEs[[iidt]][[data_name]]$MSE_test = MSE_test * 1e3
            print(MSE_test * 1e3)
        }
    }
    return(list(MSEs = MSEs, fit_object = fit_object))
}

# get data
if(!file.exists("./Data/UK_raw/data.csv")){
    NoReturns = UK_data_appregation()
    NoReturns = UK_get_dense_from_raw()
}
if(!file.exists("./Data/IID/RealData/UK/dense/T_obs.csv")){
    UK_folder = "./Data/IID/RealData/UK/"
    get_sample_from_dense(UK_folder, sparsity = c(30, 30), iidt = TRUE)
    get_sample_from_dense(UK_folder, sparsity = c(8, 12), iidt = TRUE)
    get_sample_from_dense(UK_folder, sparsity = c(30, 30), iidt = FALSE)
    get_sample_from_dense(UK_folder, sparsity = c(8, 12), iidt = FALSE)
}

# analyze the UK dataset using PACE, 1DS, and MICE
result = do_analysis()