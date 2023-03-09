rm(list = ls())
setwd("/Users/eric/Desktop/UCD/TransFD")
library(plotly); library(fdapace); library(mice)
get_design_plot = function(all_data){
    all_data = all_data_list[[1]]
    time = all_data$T_obs
    time_len = dim(all_data$X_full_noise)[2]
    result = matrix(0, time_len, time_len)
    for(i in 1:dim(time)[1]){
        now_time = round(na.omit(time[i, ]) * (time_len-1)) + 1
        result[now_time, now_time] = result[now_time, now_time] + 1
    }
    diag_result = diag(result)
    diag(result)[1] = mean(c(result[1, 2], result[2, 1]))
    for(i in 2:time_len){diag(result)[i] = mean(c(result[i - 1, i], result[i, i - 1]))}
    par(mar=c(5.1, 4.1, 4.1, 4.1)) # adapt margins
    return(list(diag_result, result))
}
myMICE = function(do_MICE, all_data, index){
    if(do_MICE){
        L = dim(all_data$X_full_noise)[2]
        X_MICE = matrix(NA, length(all_data$X_obs), L)
        for(i in 1:num_subjects){X_MICE[i, round(all_data$T_obs[[i]] * (L - 1) + 1)] = all_data$X_obs[[i]]}
        mice.fit = mice(X_MICE, m = 1, maxit = 1)
        MICE_est = as.matrix(complete(mice.fit))
        MSE_train = mean((MICE_est[index$training, ] - all_data$X_full_noise[index$training, ])^2, na.rm = TRUE)
        MSE_test = mean((MICE_est[index$testing, ] - all_data$X_full_noise[index$testing, ])^2, na.rm = TRUE)
        return(list(MSE_train = MSE_train, MSE_test = MSE_test, EST_train = MICE_est[index$training, ], EST_test = MICE_est[index$testing, ]))
    }else{
        return(list(MSE_train = NA, MSE_test = NA, EST_train = NA, EST_test = NA))
    }
    
}
my1DS = function(do_1DS, all_data, index, obs_per_sub){
    if(do_1DS){
        n_train = length(index$training)
        subsamples = ceiling((runif(n_train)/10 + 0.1) * obs_per_sub[1:n_train])
        cv_pts_list = lapply(1:n_train, FUN = function(i){sort(sample(1:obs_per_sub[i], subsamples[i]))})
        not_cv_pts_list = lapply(1:n_train, FUN = function(i){setdiff(1:obs_per_sub[i], cv_pts_list[[i]])})
        
        bw_list = 1/mean(obs_per_sub) * c(1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10)
        CV_list = bw_list * 0
        counter_1D = 1
        for(bw in bw_list){
            error_now = 0
            for(i in 1:n_train){
                cv_pts_now = cv_pts_list[[i]]
                not_cv_pts_now = not_cv_pts_list[[i]]
                Xout = Lwls1D(bw, xin = all_data$T_obs[[i]][not_cv_pts_now], yin = all_data$X_obs[[i]][not_cv_pts_now], xout = all_data$T_obs[[i]][cv_pts_now], kernel_type = "gauss")
                error_now = error_now + sum((Xout - all_data$X_obs[[i]][cv_pts_now])^2)
            }
            CV_list[counter_1D] = error_now
            counter_1D = counter_1D + 1
        }
        bw = bw_list[which.min(CV_list)]
        IDS_est = matrix(0, length(all_data$X_obs), dim(all_data$X_full_noise)[2])
        timegird = unique(sort(unlist(all_data$T_obs)))
        for(i in 1:dim(IDS_est)[1]){
            Xout = Lwls1D(bw, xin = all_data$T_obs[[i]], yin = all_data$X_obs[[i]], xout = timegird, kernel_type = "gauss")
            IDS_est[i, ] = Xout
        }
        MSE_train = mean((IDS_est[index$training, ] - all_data$X_full_noise[index$training, ])^2)
        MSE_test = mean((IDS_est[index$testing, ] - all_data$X_full_noise[index$testing, ])^2)
        return(list(MSE_train = MSE_train, MSE_test = MSE_test, EST_train = IDS_est[index$training, ], EST_test = IDS_est[index$testing, ]))
    }else{
        return(list(MSE_train = NA, MSE_test = NA, EST_train = NA, EST_test = NA))
    }
}
myPACE = function(do_PACE, all_data, index, optns = NULL, Transformer = FALSE){
    # optns = list(dataType = "Sparse", usergrid = TRUE, maxK = 20, userBwCov = 0.03)
    # all_data = Imputed_all_data; index = index; optns = list(dataType = "Sparse", usergrid = TRUE, maxK = 20, userBwCov = 0.03)
    if(do_PACE){
        num_subjects = length(all_data$X_obs)
        PACE_fit = FPCA(all_data$X_obs[index$training], all_data$T_obs[index$training], optns = optns)
        PACE_est = predict(PACE_fit, all_data$X_obs, all_data$T_obs)[[2]]
        if(Transformer){index$training = setdiff(index$training, index$testing)}
        MSE_train = mean((PACE_est[index$training, ] - all_data$X_full_noise[index$training, ])^2)
        MSE_test = mean((PACE_est[index$testing, ] - all_data$X_full_noise[index$testing, ])^2)
        return(list(MSE_train = MSE_train, MSE_test = MSE_test, EST_train = PACE_est[index$training, ], EST_test = PACE_est[index$testing, ], fit = PACE_fit))
    }else{
        return(list(MSE_train = NA, MSE_test = NA, EST_train = NA, EST_test = NA))
    }
}
get_data_from_full = function(sparsity = c(8, 12), iidt = TRUE){
    # sparsity = c(8, 12); iidt = TRUE
    set.seed(5170328)
    
    X_full_noise = as.matrix(data.table::fread(file = "./Data/IID/RealData/UK/X_full_noise.csv", header = FALSE))
    T_full = as.matrix(data.table::fread(file = "./Data/IID/RealData/UK/T_full.csv", header = FALSE))
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
    
    folder = ifelse(iidt, "./Data/IID/RealData/UK", "./Data/NonIID/RealData/UK")
    folder = ifelse(diff(sparsity) == 0, paste(folder, "/dense", sep = ""), paste(folder, "/sparse", sep = ""))
    data.table::fwrite(T_obs, file = paste(folder, "/T_obs.csv", sep = ""), row.names = FALSE, col.names = FALSE)
    data.table::fwrite(X_obs, file =  paste(folder, "/X_obs.csv", sep = ""), row.names = FALSE, col.names = FALSE)
    
    return(list(X_obs = X_obs, T_obs = T_obs, X_full_noise = X_full_noise))
}

# settings
iidt = FALSE
fast_PACE = FALSE
sparsity_list = list(c(30, 30), c(8, 12))
split = c(90, 5, 5)
do_PACE = TRUE; do_1DS = TRUE; do_MICE = TRUE; do_trans = FALSE
save_plot = FALSE
denoise_method_list = c("None", "l1w", "l2w", "TVo", "l2o")

## auxiliary
# for transformer
fitNoneTrans = fitPaceTrans = fit1DSTrans = vector(mode = "list", length = length(denoise_method_list))
names(fitNoneTrans) = names(fitPaceTrans) = names(fit1DSTrans) = denoise_method_list
# for MSE
MSE_train_MAT = matrix(0, 2, 6)
colnames(MSE_train_MAT) = c("PACE", "1DS", "mice", "Transformer", "Trans + PACE", "Trans + 1DS")
rownames(MSE_train_MAT) = c("Dense w/ noise", "Sparse w/ noise")
MSE_train = lapply(1:length(denoise_method_list), function(i){return(MSE_train_MAT)})
names(MSE_train) = denoise_method_list
MSE_test = MSE_train
gen_data = result_list = vector(mode = "list", length = 2)

# generate data
all_data_list = vector("list", 2)
counter = 1
for(sparsity in sparsity_list){
    # noise_to_singal = 0; sparsity = c(30, 30)
    all_data = get_data_from_full(sparsity = sparsity, iidt = iidt)
    all_data_list[[counter]] = all_data
    counter = counter + 1
}

dev.off()
design = get_design_plot(all_data_list[[1]])
plot(design[[1]], type = "l")
plot_ly(z = ~design[[2]], type = "contour", colors = c("white", "black"))

counter = 1
for(sparsity in sparsity_list){
    # sparsity = c(30, 30)
    
    all_data = all_data_list[[counter]] # get data
    # preprocess
    obs_per_sub = all_data$X_obs[, 1]
    all_data$X_obs = all_data$X_obs[, -1]
    num_subjects = length(obs_per_sub)
    all_data$X_obs = lapply(1:num_subjects, function(i){return(all_data$X_obs[i, 1:obs_per_sub[i]])})
    all_data$T_obs = lapply(1:num_subjects, function(i){return(all_data$T_obs[i, 1:obs_per_sub[i]])})
    gen_data[[counter]] = all_data
    
    n_train = round(num_subjects * sum(split[1:2])/sum(split))
    index = list(training = 1:n_train, testing = (n_train + 1):num_subjects)
    
    # PACE
    maxK = 49
    optns = list(dataType = "Sparse", usergrid = TRUE, maxK = maxK, methodBwCov = "GCV", plot = TRUE)
    if(fast_PACE){optns = list(dataType = "Sparse", usergrid = TRUE, maxK = maxK, userBwCov = 0.03, plot = TRUE)}
    fitPACE = myPACE(do_PACE, all_data, index, optns)
    
    # 1D smoother
    fit1DS = my1DS(do_1DS, all_data, index, obs_per_sub)
    
    # MICE
    fitMICE = myMICE(do_MICE, all_data, index)
    
    ## Transformer
    dw_folder = paste(ifelse(sparsity[1] == sparsity[2], "dense", "sparse"))
    # i = 1
    for(i in 1:length(denoise_method_list)){
        print(c(counter, i))
        if(do_trans){
            denoise_method = denoise_method_list[i]
            folder = ifelse(iidt, "./real_data", "./real_data_frag")
            ImputedX = as.matrix(read.csv(paste(folder, "/TransImputed/Vanilla/", dw_folder, "/X_imputed_", denoise_method, ".csv", sep = ""), header=FALSE))
            timegird = unique(sort(unlist(all_data$T_obs)))
            Imputed_all_data = list(X_true = all_data$X_true, X_full_noise = all_data$X_full_noise, X_obs = lapply(1:num_subjects, function(i){return(ImputedX[i, ])}), T_obs = lapply(1:num_subjects, function(i){return(timegird)}))
            
            # None
            MSE_trans_train = mean((ImputedX[index$training, ] - all_data$X_true[index$training, ])^2)
            MSE_trans_test = mean((ImputedX[index$testing, ] - all_data$X_true[index$testing, ])^2)
            MSE_trans_train = c(MSE_trans_train, mean((ImputedX[index$training, ] - all_data$X_full_noise[index$training, ])^2))
            MSE_trans_test = c(MSE_trans_test, mean((ImputedX[index$testing, ] - all_data$X_full_noise[index$testing, ])^2))
            fitNoneTrans[[i]] = list(MSE_train = MSE_trans_train, MSE_test = MSE_trans_test, EST_train = ImputedX[index$training, ], EST_test = ImputedX[index$testing, ])
        }else{
            Imputed_all_data = NULL
            fitNoneTrans[[i]] = list(MSE_train = NA, MSE_test = NA, EST_train = NA, EST_test = NA)
        }
        
        # PACE
        indexTrans = list(training = 1:num_subjects, testing = index$testing)
        optns = list(dataType = "Sparse", usergrid = TRUE, maxK = maxK, methodBwCov = "GCV")
        if(fast_PACE){optns = list(dataType = "Sparse", usergrid = TRUE, maxK = maxK, userBwCov = 0.01)}
        fitPaceTrans[[i]] = myPACE(do_trans, all_data = Imputed_all_data, indexTrans, optns, Transformer = TRUE)
        
        # 1DS
        fit1DSTrans[[i]] = my1DS(do_trans, all_data = Imputed_all_data, index, obs_per_sub = rep(dim(all_data$X_true)[2], dim(all_data$X_true)[1]))
        
        # error matrix
        MSE_train[[i]][counter, ] = cbind(fitPACE$MSE_train, fit1DS$MSE_train, fitMICE$MSE_train, fitNoneTrans[[i]]$MSE_train, fitPaceTrans[[i]]$MSE_train, fit1DSTrans[[i]]$MSE_train)[1, ]
        MSE_test[[i]][counter, ] = cbind(fitPACE$MSE_test, fit1DS$MSE_test, fitMICE$MSE_test, fitNoneTrans[[i]]$MSE_test, fitPaceTrans[[i]]$MSE_test, fit1DSTrans[[i]]$MSE_test)[1, ]
    }
    result_list[[counter]] = list(fitPACE = fitPACE, fit1DS = fit1DS, fitMICE = fitMICE, fitNoneTrans = fitNoneTrans, fitPaceTrans = fitPaceTrans, fit1DSTrans = fit1DSTrans)
    counter = counter + 1
}

mul = 1e3
MSE_train = lapply(MSE_train, function(x){x * mul})
MSE_test = lapply(MSE_test, function(x){x * mul})

# save gen_data/result_list/matrices
MSE_list = list(MSE_train = MSE_train, MSE_test = MSE_test)
saveRDS(MSE_list, paste(ifelse(iidt, "./R_result/IID/RealData", "./R_result/NonIID/RealData"), "/MSE_list.rds", sep = ""))
saveRDS(gen_data, paste(ifelse(iidt, "./R_result/IID/RealData", "./R_result/NonIID/RealData"), "/gen_data.rds", sep = ""))
saveRDS(result_list, paste(ifelse(iidt, "./R_result/IID/RealData", "./R_result/NonIID/RealData"), "/result_list.rds", sep = ""))

# read 
setwd("/Users/eric/Desktop/UCD/TransFD/")
MSE_list = readRDS(paste(ifelse(iidt, "./R_result/IID/RealData", "./R_result/NonIID/RealData"), "/MSE_list.rds", sep = ""))
gen_data = readRDS(paste(ifelse(iidt, "./R_result/IID/RealData", "./R_result/NonIID/RealData"), "/gen_data.rds", sep = ""))
result_list = readRDS(paste(ifelse(iidt, "./R_result/IID/RealData", "./R_result/NonIID/RealData"), "/result_list.rds", sep = ""))
sparsity_list = list(c(30, 30), c(8, 12))

# this part is for output
MSE_list2 = MSE_list
for(i in 1:length(MSE_list)){
    MSE_MAT = MSE_list[[i]]
    for(j in 1:length(MSE_MAT)){
        tmp = MSE_MAT[[j]][, c(1, 2, 4, 5, 6)]
        colnames(tmp) = c("PACE", "1DS", "Vanilla", "Trans+PACE", "Trans+1DS")
        MSE_list2[[i]][[j]] = tmp
    }
}
MSE_list2$MSE_test

# save data imputed by PACE
counter = 1
for(sparsity in sparsity_list){
    # sparsity = c(30, 30)
    ImputedX = result_list[[counter]]$fitPaceTrans$None$EST_train
    dw_folder = ifelse(sparsity[1] == sparsity[2], "dense", "sparse")
    file = paste("./data/PaceImputed", dw_folder, "X_imputed_None.csv", sep = "/")
    data.table::fwrite(ImputedX, file = file, row.names = FALSE, col.names = FALSE)
    counter = counter + 1
}

# plot
do_MICE = FALSE
index_training = seq(0, n_train, floor(n_train/500)) + 1
index_testing = (n_train + 1):num_subjects
index_all = unique(c(index_training, index_testing))
save_plot = TRUE
timegrid = pracma::linspace(0, 1, 96)
counter = 1
for(sparsity in sparsity_list){
    # noise_to_singal = 0; sparsity = c(30, 30)
    DN = ifelse(sparsity[2] - sparsity[1] == 0, "dense", "sparse")
    if(save_plot){foldername = ifelse(iidt, paste("/Users/eric/Desktop/UCD/TransFD/plots/RealData/investigate/", DN, sep = ""), paste("/Users/eric/Desktop/UCD/TransFD/plots_frag/RealData/investigate/", DN, sep = ""))}
    all_data = gen_data[[counter]]
    fits = result_list[[counter]]
    for(counter_in in 1:length(index_all)){
        i = index_all[counter_in]
        testInd = i - n_train
        if(counter_in %% 100 == 0 && testInd > 0){
            if(save_plot){
                filename = paste("/", counter_in, "Rplot.png", sep = "")
                png(file = paste(foldername, filename, sep = ""), width = 640, height = 490)
                par(mar = c(3, 9, 2, 2))
            }
            plot(timegrid, all_data$X_full_noise[i, ], type = 'l', ylim = c(-1, 3), lwd = 3, col = "orange", xlab = "", ylab = "")
            title(line = 2, ylab = "PACE/1DS", cex.lab = 4)
            if(do_PACE){lines(timegrid, fits$fitPACE$EST_test[testInd, ], col = 'red', lwd = 3)}
            if(do_1DS){lines(timegrid, fits$fit1DS$EST_test[testInd, ], col = 'green', lwd = 3)}
            if(do_MICE){lines(which(!is.na(fits$fitMICE$EST_test[testInd, ]))/100 - 0.01, fits$fitMICE$EST_test[testInd, which(!is.na(fits$fitMICE$EST_test[testInd, ]))], col = 'purple', lwd = 3)}
            points(all_data$T_obs[[i]], all_data$X_obs[[i]], col = "blue", pch = 16, cex = 1.5)
            if(!do_MICE & save_plot){legend("topright", legend = c("FPCA", "1DS", "true", "obs"), col = c("red", "green", "orange", "blue"), lty = c(1, 1, 1, NA), pch = c(NA, NA, NA, 16), cex = 2)}
            if(do_MICE & save_plot){legend("topright", legend = c("FPCA", "1DS", "mice", "true", "obs"), col = c("red", "green", "purple", "orange", "blue"), lty = c(1, 1, 1, 1, NA), pch = c(NA, NA, NA, NA, 16), cex = 2)}
            if(save_plot){dev.off()}
            if(!save_plot){Sys.sleep(0.1)}
            
            if(do_trans){
                if(save_plot){
                    # filename = paste("/RplotTrans", i - 1, ".png", sep = "")
                    filename = paste("/", i, "RplotTrans.png", sep = "")
                    png(file = paste(foldername, filename, sep = ""), width = 625, height = 490)
                    par(mar = c(3, 9, 2, 2))
                }
                plot(timegrid, all_data$X_true[i + n_train, ], type = 'l', ylim = c(-7, 7), lwd = 3, col = "orange", xlab = "", ylab = "")
                title(line = 2, ylab = "l2o+Polishers", cex.lab = 4)
                if(i == 1){plot(timegrid, all_data$X_true[i + n_train, ], type = 'l', ylim = c(-7, 7), lwd = 3, col = "orange", xlab = "", ylab = DN, cex.lab = 4)}
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

MSE_list$MSE_test
result_list[[1]]$fitPACE$fit$selectK
result_list[[2]]$fitPACE$fit$selectK


