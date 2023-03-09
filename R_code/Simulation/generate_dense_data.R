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
