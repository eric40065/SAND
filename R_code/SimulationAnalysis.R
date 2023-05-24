#setwd("./SAND")
library(plotly); library(fdapace); library(mice); library(stringr)
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
generate_dense_data = function(num_subjects, num_basis = 25, noise_to_singal = 0.25, score_dist = "G"){
    # num_subjects = 10000
    # num_basis = 20
    # noise_to_singal = 0.25
    # score_dist = "G"
    if(!score_dist %in% c("G", "T", "E")){stop("Wrong score distribution")}
    set.seed(5170223)
    save_folder = paste("./Data/IID/Simulation", ifelse(mean(num_basis) < 10, "/LowDim_", ifelse(mean(num_basis) < 14, "/Hybrid_", "/HighDim_")), score_dist, sep = "")
    t = pracma::linspace(0, 1, 101)
    max_basis = max(num_basis)
    t_mat = matrix(t, nrow = max_basis, ncol = length(t), byrow = T)
    k_mat = matrix(1:max_basis, nrow = max_basis, ncol = length(t))
    
    phi1 = sin(2 * k_mat * pi * t_mat) / (k_mat)
    phi2 = cos(2 * k_mat * pi * t_mat) / (k_mat)
    
    if(length(num_basis) > 1){
        basis_per_sub = sample(num_basis[1]:num_basis[2], 1e4, replace = TRUE, prob = dgamma(num_basis[1]:num_basis[2], 3, scale = 2))
        # hist(basis_per_sub, breaks = num_basis[1]:num_basis[2])
        mask_mat = matrix(0, num_subjects, max_basis)
        for(i in 1:num_subjects){mask_mat[i, 1:basis_per_sub[i]] = 1}
        data.table::fwrite(as.matrix(basis_per_sub), file = paste(save_folder, "/basis_per_sub.csv", sep = ""), row.names = FALSE, col.names = FALSE)
        if(score_dist == "G"){
            a_mat = matrix(rnorm(num_subjects * max_basis), num_subjects, max_basis) * mask_mat
            b_mat = matrix(rnorm(num_subjects * max_basis), num_subjects, max_basis) * mask_mat
        }else if(score_dist == "E"){
            a_mat = matrix(rexp(num_subjects * max_basis, 1) - 1, num_subjects, max_basis) * mask_mat
            b_mat = matrix(rexp(num_subjects * max_basis, 1) - 1, num_subjects, max_basis) * mask_mat
        }else{
            a_mat = matrix(rt(num_subjects * max_basis, 5), num_subjects, max_basis) * mask_mat
            b_mat = matrix(rt(num_subjects * max_basis, 5), num_subjects, max_basis) * mask_mat
        }
    }else{
        if(score_dist == "G"){
            a_mat = matrix(rnorm(num_subjects * max_basis), num_subjects, num_basis)
            b_mat = matrix(rnorm(num_subjects * max_basis), num_subjects, num_basis)
        }else if(score_dist == "E"){
            a_mat = matrix(rexp(num_subjects * num_basis, 1) - 1, num_subjects, num_basis)
            b_mat = matrix(rexp(num_subjects * num_basis, 1) - 1, num_subjects, num_basis)
        }else{
            a_mat = matrix(rt(num_subjects * num_basis, 5), num_subjects, num_basis)
            b_mat = matrix(rt(num_subjects * num_basis, 5), num_subjects, num_basis)
        }
    }
    
    X_dense_true = a_mat %*% phi1 + b_mat %*% phi2
    data.table::fwrite(X_dense_true, file = paste(save_folder, "/X_full_true.csv", sep = ""), row.names = FALSE, col.names = FALSE)
    
    X_sd = mean(apply(X_dense_true, 2, sd))
    e_sd = X_sd * noise_to_singal
    ME = matrix(rnorm(num_subjects * length(t), sd = e_sd), num_subjects, length(t))
    
    X_dense = X_dense_true + ME
    T_dense = matrix(t, nrow = num_subjects, ncol = length(t), byrow = T)
    
    data.table::fwrite(X_dense, file = paste(save_folder, "/X_full_noise.csv", sep = ""), row.names = FALSE, col.names = FALSE)
    data.table::fwrite(T_dense, file = paste(save_folder, "/T_full.csv", sep = ""), row.names = FALSE, col.names = FALSE)
    
    return(save_folder)
}
get_all_data = function(num_subjects = 1e4, 
                        num_basis_list = c(5, 20),
                        iidt_list = c(TRUE, FALSE),
                        sparsity_list = list(c(30, 30), c(8, 12)), 
                        noise_to_singal_list = c(0.25, 0), 
                        score_dist_list = c("G", "E", "T")){
    # num_subjects = 1e4; num_basis_list = c(3, 20); iidt_list = c(TRUE, FALSE); sparsity_list = list(c(30, 30), c(8, 12)); noise_to_singal_list = c(0.25, 0); score_dist_list = c("G", "E", "T")
    # generate data
    for(num_basis in num_basis_list){
        for(score_dist in score_dist_list){
            folder = generate_dense_data(num_subjects, num_basis, noise_to_singal_list[1], score_dist = score_dist)
            for(iidt in iidt_list){
                for(noise_to_singal in noise_to_singal_list){
                    for(sparsity in sparsity_list){
                        generate_from_data(folder, sparsity = sparsity, noise_to_singal = noise_to_singal, iidt = iidt)
                    }
                }
            }
        }
    }
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
do_analysis = function(data_name_list = c("LowDim_G", "LowDim_T", "LowDim_E", "HighDim_G", "HighDim_T", "HighDim_E"), iidt_list = c("IID", "NonIID"),
                       dense_error_list = list(c("dense", "w_error"), c("sparse", "w_error"), c("dense", "wo_error"), c("sparse", "wo_error")),
                       split = c(90, 5, 5), do_PACE = TRUE, do_1DS = TRUE, do_MICE = TRUE, do_trans = FALSE, save_plot = FALSE,
                       denoise_method_list = c("l1w", "l2w"), fast = FALSE){
    # data_name_list = c("LowDim_G", "LowDim_T", "LowDim_E", "HighDim_G", "HighDim_T", "HighDim_E"); iidt_list = c("IID", "NonIID");
    # dense_error_list = list(c("dense", "w_error"), c("sparse", "w_error"), c("dense", "wo_error"), c("sparse", "wo_error"));
    # split = c(90, 5, 5); do_PACE = FALSE; do_1DS = FALSE; do_MICE = FALSE; do_trans = TRUE; save_plot = FALSE;
    # denoise_method_list = c("None", "l1w", "l2w", "TVo", "l2o"); fast = FALSE;
    
    fit_object_3 = list(fitPACE = NA, fit1DS = NA, fitMICE = NA)
    fit_object_2 = setNames(rep(list(fit_object_3), length(dense_error_list)), sapply(dense_error_list, function(i){paste(i[1], i[2], sep = "_")}))
    fit_object_1 = setNames(rep(list(fit_object_2), length(data_name_list)), data_name_list)
    fit_object = setNames(rep(list(fit_object_1), length(iidt_list)), iidt_list)
    
    fit_object_1 = setNames(rep(list(list(MSE_train = NA, MSE_test = NA, MSE_train_noise = NA, MSE_test_noise = NA)), length(data_name_list)), data_name_list)
    MSEs = setNames(rep(list(fit_object_1), length(iidt_list)), iidt_list)
    # iidt = "IID"; data_name = "HighDim_E"; dense_error = dense_error_list[[2]]
    for(iidt in iidt_list){
        true_folder = "./Data/IID/Simulation"
        obs_folder  = paste("./Data", iidt, "Simulation", sep = "/")
        for(data_name in data_name_list){
            true_datafolder = paste(true_folder, data_name, sep = "/")
            obs_datafolder  = paste(obs_folder,  data_name, sep = "/")
            
            # get data in full grid
            X_full_true  = as.matrix(data.table::fread(file = paste(true_datafolder, "/X_full_true.csv", sep = ""), header = FALSE))
            X_full_noise = as.matrix(data.table::fread(file = paste(true_datafolder, "/X_full_noise.csv", sep = ""), header = FALSE))
            
            ## auxiliary
            # MSE
            if(data_name == "HighDim_E" && do_trans){
                fitNoneTrans = fitPaceTrans = fit1DSTrans = vector(mode = "list", length = length(denoise_method_list))
                names(fitNoneTrans) = names(fitPaceTrans) = names(fit1DSTrans) = denoise_method_list
                MSE_train = matrix(0, 3 + 3 * length(denoise_method_list), 4)
                colnames(MSE_train) = c("Dense w/ noise", "Sparse w/ noise", "Dense wo/ noise", "Sparse wo/ noise")
                rownames(MSE_train) = c("Pace", "1DS", "Mice", as.vector(sapply(denoise_method_list, function(method){return(paste(method, c("", " + Pace", " + 1DS"), sep = ""))})))
                MSE_test_noise = MSE_train_noise = MSE_test = MSE_train
            }else{
                MSE_train = matrix(0, 3, 4)
                rownames(MSE_train) = c("Pace", "1DS", "Mice")
                colnames(MSE_train) = c("Dense w/ noise", "Sparse w/ noise", "Dense wo/ noise", "Sparse wo/ noise")
                MSE_test_noise = MSE_train_noise = MSE_test = MSE_train
            }
            counter_col = 0
            for(dense_error in dense_error_list){# aaa}}}
                # dense_error = dense_error_list[[1]]
                print(iidt)
                print(data_name)
                print(dense_error)
                counter_col = counter_col + 1
                
                # get data
                obs_datafolder_de = paste(obs_datafolder, dense_error[1], dense_error[2], sep = "/")
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
                optns = list(dataType = "Sparse", usergrid = TRUE, maxK = 49, methodBwCov = "GCV", plot = TRUE)
                if(fast){optns = list(dataType = "Sparse", usergrid = TRUE, maxK = 1, userBwCov = 0.03, plot = TRUE)}
                fitPACE = myPACE(do_PACE, X_obs, T_obs, X_full_true, X_full_noise, index, optns)
                MSE_train      [1, counter_col] = fitPACE$MSE_train[1]
                MSE_train_noise[1, counter_col] = fitPACE$MSE_train[2]
                MSE_test       [1, counter_col] = fitPACE$MSE_test[1]
                MSE_test_noise [1, counter_col] = fitPACE$MSE_test[2]
                fit_object[[iidt]][[data_name]][[counter_col]]$fitPACE = fitPACE
                
                # 1D smoother
                fit1DS = my1DS(do_1DS, X_obs, T_obs, X_full_true, X_full_noise, index, obs_per_sub)
                MSE_train      [2, counter_col] = fit1DS$MSE_train[1]
                MSE_train_noise[2, counter_col] = fit1DS$MSE_train[2]
                MSE_test       [2, counter_col] = fit1DS$MSE_test[1]
                MSE_test_noise [2, counter_col] = fit1DS$MSE_test[2]
                fit_object[[iidt]][[data_name]][[counter_col]]$fit1DS = fit1DS
                
                # MICE
                fitMICE = myMICE(do_MICE, X_obs, T_obs, X_full_true, X_full_noise, index, fast)
                MSE_train      [3, counter_col] = fitMICE$MSE_train[1]
                MSE_train_noise[3, counter_col] = fitMICE$MSE_train[2]
                MSE_test       [3, counter_col] = fitMICE$MSE_test[1]
                MSE_test_noise [3, counter_col] = fitMICE$MSE_test[2]
                fit_object[[iidt]][[data_name]][[counter_col]]$fitMICE = fitMICE
                
                ## Transformer
                if(data_name == "HighDim_E" && do_trans){
                    counter_row = 3
                    fit_object_3 = list(fitNoneTrans = NA, fitPaceTrans = NA, fit1DSTrans = NA)
                    fit_object_2 = setNames(rep(list(fit_object_3), length(denoise_method_list)), denoise_method_list)
                    fit_object[[iidt]][[data_name]][[counter_col]] = append(fit_object[[iidt]][[data_name]][[counter_col]], fit_object_2)
                    
                    true_index = list(training = seq(1, 9500, 1), testing = 9501:10000) # Full analysis
                    # true_index = list(training = seq(1, 9500, 19), testing = 9501:10000) # for a faster result
                    trans_X_full_true = X_full_true[c(true_index[[1]], true_index[[2]]), ]
                    trans_X_full_noise = X_full_noise[c(true_index[[1]], true_index[[2]]), ]
                    trans_index = list(training = 1:length(true_index$training), testing = length(true_index$training) + (1:length(true_index$testing))) # for a faster result
                    for(i in 1:length(denoise_method_list)){
                        # i = 1
                        denoise_method = denoise_method_list[i]
                        print(denoise_method)
                        Imputation_folder = paste(str_replace(obs_folder, "Data", "ImputedData"), data_name, "Vanilla", dense_error[1], dense_error[2], sep = "/")
                        ImputedX = as.matrix(read.csv(paste(Imputation_folder, "/X_imputed_", denoise_method, ".csv", sep = ""), header = FALSE))
                        timegird = unique(sort(unlist(T_obs)))
                        ImputedX_list = lapply(1:dim(ImputedX)[1], function(i){return(ImputedX[i, ])})
                        ImputedT_list = lapply(1:dim(ImputedX)[1], function(i){return(timegird)})
                        
                        # None
                        MSE_trans_train = mean((ImputedX[trans_index$training, ] - trans_X_full_true[trans_index$training, ])^2)
                        MSE_trans_test = mean((ImputedX[trans_index$testing, ] - trans_X_full_true[trans_index$testing, ])^2)
                        MSE_trans_train = c(MSE_trans_train, mean((ImputedX[trans_index$training, ] - trans_X_full_noise[trans_index$training, ])^2))
                        MSE_trans_test = c(MSE_trans_test, mean((ImputedX[trans_index$testing, ] - trans_X_full_noise[trans_index$testing, ])^2))
                        fitNoneTrans[[i]] = list(MSE_train = MSE_trans_train, MSE_test = MSE_trans_test, EST_train = ImputedX[trans_index$training, ], EST_test = ImputedX[trans_index$testing, ])
                        MSE_train      [counter_row + 1, counter_col] = fitNoneTrans[[i]]$MSE_train[1]
                        MSE_train_noise[counter_row + 1, counter_col] = fitNoneTrans[[i]]$MSE_train[2]
                        MSE_test       [counter_row + 1, counter_col] = fitNoneTrans[[i]]$MSE_test[1]
                        MSE_test_noise [counter_row + 1, counter_col] = fitNoneTrans[[i]]$MSE_test[2]
                        fit_object[[iidt]][[data_name]][[counter_col]][[denoise_method]]$fitNoneTrans = fitNoneTrans
                        
                        # PACE
                        optns = list(dataType = "Sparse", usergrid = TRUE, maxK = 49, methodBwCov = "GCV")
                        if(fast){optns = list(dataType = "Sparse", usergrid = TRUE, maxK = 1, userBwCov = 0.01, plot = TRUE)}
                        fitPaceTrans[[i]] = myPACE(do_trans, ImputedX_list, ImputedT_list, trans_X_full_true, trans_X_full_noise, trans_index, optns, Transformer = TRUE)
                        MSE_train      [counter_row + 2, counter_col] = fitPaceTrans[[i]]$MSE_train[1]
                        MSE_train_noise[counter_row + 2, counter_col] = fitPaceTrans[[i]]$MSE_train[2]
                        MSE_test       [counter_row + 2, counter_col] = fitPaceTrans[[i]]$MSE_test[1]
                        MSE_test_noise [counter_row + 2, counter_col] = fitPaceTrans[[i]]$MSE_test[2]
                        fit_object[[iidt]][[data_name]][[counter_col]][[denoise_method]]$fitPaceTrans = fitPaceTrans
                        
                        # 1DS
                        fit1DSTrans[[i]] = my1DS(do_trans, ImputedX_list, ImputedT_list, trans_X_full_true, trans_X_full_noise, trans_index, obs_per_sub = rep(length(timegird), dim(ImputedX)[1]))
                        MSE_train      [counter_row + 3, counter_col] = fit1DSTrans[[i]]$MSE_train[1]
                        MSE_train_noise[counter_row + 3, counter_col] = fit1DSTrans[[i]]$MSE_train[2]
                        MSE_test       [counter_row + 3, counter_col] = fit1DSTrans[[i]]$MSE_test[1]
                        MSE_test_noise [counter_row + 3, counter_col] = fit1DSTrans[[i]]$MSE_test[2]
                        fit_object[[iidt]][[data_name]][[counter_col]][[denoise_method]]$fit1DSTrans = fit1DSTrans
                        counter_row = counter_row + 3
                    }
                }
            }
            MSEs[[iidt]][[data_name]]$MSE_train = MSE_train * 1e3
            MSEs[[iidt]][[data_name]]$MSE_test = MSE_test * 1e3
            MSEs[[iidt]][[data_name]]$MSE_train_noise = MSE_train_noise * 1e3
            MSEs[[iidt]][[data_name]]$MSE_test_noise = MSE_test_noise * 1e3
            print(MSE_test * 1e3)
        }
    }
    return(list(MSEs = MSEs, fit_object = fit_object))
}
plot_result = function(result){
    # data_name_list = c("LowDim_G", "LowDim_T", "LowDim_E", "HighDim_G", "HighDim_T", "HighDim_E"); iidt_list = c("IID", "NonIID");
    # dense_error_list = list(c("dense", "w_error"), c("sparse", "w_error"), c("dense", "wo_error"), c("sparse", "wo_error"));
    # split = c(90, 5, 5); denoise_method_list = c("None", "l1w", "l2w", "TVo", "l2o")
    # iidt = iidt_list[1]; data_name = data_name_list[1]; dense_error = dense_error_list[[1]]; counter_in = length(index_all)
    
    for(iidt in iidt_list){
        true_folder = "./Data/IID/Simulation"
        obs_folder  = paste("./Data", iidt, "Simulation", sep = "/")
        
        for(data_name in data_name_list){
            true_datafolder = paste(true_folder, data_name, sep = "/")
            obs_datafolder  = paste(obs_folder,  data_name, sep = "/")
            
            # get data in full grid
            X_full_true  = as.matrix(data.table::fread(file = paste(true_datafolder, "/X_full_true.csv", sep = ""), header = FALSE))
            X_full_noise = as.matrix(data.table::fread(file = paste(true_datafolder, "/X_full_noise.csv", sep = ""), header = FALSE))
            
            counter_col = 0
            for(dense_error in dense_error_list){# aaa}}}
                counter_col = counter_col + 1
                
                # get data
                fits = result$fit_object[[iidt]][[data_name]][[counter_col]]
                obs_datafolder_de = paste(obs_datafolder, dense_error[1], dense_error[2], sep = "/")
                X_obs = as.matrix(data.table::fread(file = paste(obs_datafolder_de, "/X_obs.csv", sep = ""), header = FALSE))
                T_obs = as.matrix(data.table::fread(file = paste(obs_datafolder_de, "/T_obs.csv", sep = ""), header = FALSE))
                
                # preprocessing
                num_subjects = dim(X_obs)[1]; n_train = round(num_subjects * sum(split[1:2])/sum(split))
                index = list(training = 1:n_train, testing = (n_train + 1):num_subjects)
                obs_per_sub = X_obs[, 1]
                X_obs = X_obs[, -1]
                X_obs = lapply(1:num_subjects, function(i){return(X_obs[i, 1:obs_per_sub[i]])})
                T_obs = lapply(1:num_subjects, function(i){return(T_obs[i, 1:obs_per_sub[i]])})
                
                # plot
                index_training = seq(0, n_train, n_train/500) + 1; index_testing = (n_train + 2):num_subjects
                index_all = c(index_training, index_testing)
                
                save_plot = FALSE
                timegrid = sort(unique(unlist(T_obs)))
                foldername = paste(str_replace(obs_datafolder, "Data", "Plots"), "ImputedCurves", dense_error[1], dense_error[2], sep = "/")
                for(counter_in in 1:length(index_all)){
                    i = index_all[counter_in]
                    testInd = i - n_train
                    if(counter_in %% 100 == 0 && testInd > 0){
                        if(save_plot){
                            filename = paste("/", counter_in, "Rplot.png", sep = "")
                            png(file = paste(foldername, filename, sep = ""), width = 640, height = 490)
                            par(mar = c(3, 9, 2, 2))
                        }
                        plot(timegrid, X_full_true[i, ], type = 'l', ylim = c(-7, 7), lwd = 3, col = "orange", xlab = "", ylab = "")
                        title(line = 2, ylab = "PACE/1DS/MICE", cex.lab = 4)
                        lines(timegrid, fits$fitPACE$EST_test[testInd, ], col = 'red', lwd = 3)
                        lines(timegrid, fits$fit1DS$EST_test[testInd, ], col = 'green', lwd = 3)
                        lines(timegrid, fits$fitMICE$EST_test[testInd, ], col = 'purple', lwd = 3)
                        points(T_obs[[i]], X_obs[[i]], col = "blue", pch = 16, cex = 1.5)
                        if(save_plot){legend("topright", legend = c("FPCA", "1DS", "mice", "true", "obs"), col = c("red", "green", "purple", "orange", "blue"), lty = c(1, 1, 1, 1, NA), pch = c(NA, NA, NA, NA, 16), cex = 2)}
                        if(save_plot){dev.off()}
                        if(!save_plot){Sys.sleep(0.1)}
                        
                        if(do_trans){
                            if(save_plot){
                                # filename = paste("/RplotTrans", i - 1, ".png", sep = "")
                                filename = paste("/", i, "RplotTrans.png", sep = "")
                                png(file = paste(foldername, filename, sep = ""), width = 625, height = 490)
                                par(mar = c(3, 9, 2, 2))
                            }
                            plot(timegrid, all_data$X_full_true[i + n_train, ], type = 'l', ylim = c(-7, 7), lwd = 3, col = "orange", xlab = "", ylab = "")
                            title(line = 2, ylab = "l2o+Polishers", cex.lab = 4)
                            if(i == 1){plot(timegrid, all_data$X_full_true[i + n_train, ], type = 'l', ylim = c(-7, 7), lwd = 3, col = "orange", xlab = "", ylab = DN, cex.lab = 4)}
                            lines(timegrid, fits$fitNoneTrans$None$EST_test[i, ], col = 'grey', lwd = 3)
                            lines(timegrid, fits$fit1DSTrans$l2o$EST_test[i, ], col = 'green', lwd = 3)
                            lines(timegrid, fits$fitNoneTrans$l2o$EST_test[i, ], col = 'black', lwd = 3)
                            lines(timegrid, fits$fitPaceTrans$l2o$EST_test[i, ], col = 'red', lwd = 3)
                            points(all_data$T_obs[[i + n_train]], all_data$X_obs[[i + n_train]], col = "blue", pch = 16, cex = 1.5)
                            if(save_plot){legend("topright", legend = c("None", "l2o", "l2oPACE", "l2o1DS", "true", "obs"), col = c("grey", "black", "red", "green", "orange", "blue"), lty = c(1, 1, 1, 1, 1, NA), pch = c(NA, NA, NA, NA, NA, 16), cex = 2)}
                            if(save_plot){dev.off()}
                        }
                        
                        if(!save_plot){Sys.sleep(0.5)}
                    }
                }
                counter = counter + 1
            }
        }
    }
    return(list(MSEs = MSEs, fit_object = fit_object))
}
print_table = function(table, type = "1"){
    # table = result$MSEs$IID$HighDim_E$MSE_test
    powers = floor(log10(table))
    round_table = round(table * 10^(3 - powers))/10^(3 - powers)
    pre_text = c("\\multicolumn{2}{c}{FPCA}                                   ",
                 "\\multicolumn{2}{c}{1DS}                                    ",
                 "\\multicolumn{2}{c}{Mice}                                   ")
    if(type != 1){
        pre_text = c("FPCA        ", "1DS         ", "Mice        ")
    }
    for(i in 1:dim(table)[1]){
        pracma::fprintf(pre_text[i])
        pracma::fprintf(paste("&", paste(round_table[i, ], sep = "", collapse = "  & "), " \\\\"))
        pracma::fprintf("\n")
    }
}

# Check if simulated dataset is created.
if(!file.exists("./Data/IID/Simulation/HighDim_E/dense/w_error/T_obs.csv")){get_all_data()}

# Get Table 1 -- baseline from the main text
result1 = do_analysis(iidt_list = c("IID", "NonIID"), data_name_list = "HighDim_E")

# Get Table 1 -- Transformer with penalizer from the main text
result2 = do_analysis(iidt_list = c("IID", "NonIID"), data_name_list = "HighDim_E",
                     do_PACE = FALSE, do_1DS = FALSE, do_MICE = FALSE, do_trans = TRUE)

# Get tables from the supplement (only baseline methods)
result3 = do_analysis(iidt_list = c("IID"), data_name_list = c("LowDim_G", "LowDim_T", "LowDim_E", "HighDim_G", "HighDim_T"))


# print("---------------------------------------------------------------")
# print_table(result$MSEs$IID$HighDim_E$MSE_test)
# print_table(result$MSEs$NonIID$HighDim_E$MSE_test)
# print("---------------------------------------------------------------")
# print_table(result$MSEs$IID$HighDim_T$MSE_test, 2)
# print_table(result$MSEs$NonIID$HighDim_T$MSE_test, 2)
# print("---------------------------------------------------------------")
# print_table(result$MSEs$IID$HighDim_G$MSE_test, 2)
# print_table(result$MSEs$NonIID$HighDim_G$MSE_test, 2)
# 
# 
# print("---------------------------------------------------------------")
# print_table(result$MSEs$IID$LowDim_E$MSE_test, 2)
# print_table(result$MSEs$NonIID$LowDim_E$MSE_test, 2)
# print("---------------------------------------------------------------")
# print_table(result$MSEs$IID$LowDim_T$MSE_test, 2)
# print_table(result$MSEs$NonIID$LowDim_T$MSE_test, 2)
# print("---------------------------------------------------------------")
# print_table(result$MSEs$IID$LowDim_G$MSE_test, 2)
# print_table(result$MSEs$NonIID$LowDim_G$MSE_test, 2)
