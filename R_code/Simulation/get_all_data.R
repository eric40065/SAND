get_all_data = function(num_subjects = 1e4, 
                        num_basis_list = c(5, 20),
                        iidt_list = c(TRUE, FALSE),
                        sparsity_list = list(c(30, 30), c(8, 12)), 
                        noise_to_singal_list = c(0.25, 0), 
                        score_dist_list = c("G", "E", "T")){
    # num_subjects = 1e4; num_basis_list = c(3, 20); iidt_list = c(TRUE, FALSE); sparsity_list = list(c(30, 30), c(8, 12)); noise_to_singal_list = c(0.25, 0); score_dist_list = c("G", "E", "T")
    source("./R_code/Simulation/generate_from_data.R")
    source("./R_code/Simulation/generate_dense_data.R")
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
