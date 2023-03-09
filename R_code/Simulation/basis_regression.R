rm(list = ls())
setwd("/Users/eric/Desktop/UCD/TransFD/")
X = as.matrix(read.csv(paste("./data_frag/TransImputed/DiffSelfAtt/dense/w_error/X_imputed.csv", sep = ""), header=FALSE))
Tobs = as.matrix(read.csv(paste("./data_frag/dense/w_error/T_sparse.csv", sep = ""), header=FALSE))
Xobs = as.matrix(read.csv(paste("./data_frag/dense/w_error/X_sparse.csv", sep = ""), header=FALSE))
Xobs = Xobs[, 2:dim(Xobs)[2]]
Xtrue = as.matrix(read.csv("./data/X_dense_true.csv", header=FALSE))

t = seq(0, 1, 0.01)
designXfull = data.frame(X1 = sin(2 * pi * t), X2 = sin(4 * pi * t), X3 = sin(6 * pi * t), X4 = sin(8 * pi * t),
                     X5 = sin(10 * pi * t), X6 = sin(12 * pi * t), X7 = sin(14 * pi * t), X8 = sin(16 * pi * t),
                     Y1 = cos(2 * pi * t), Y2 = cos(4 * pi * t), Y3 = cos(6 * pi * t), Y4 = cos(8 * pi * t),
                     Y5 = cos(10 * pi * t), Y6 = cos(12 * pi * t), Y7 = cos(14 * pi * t), Y8 = cos(16 * pi * t))
# designXfull = designXfull/matrix(c(1:8, 1:8), nrow = dim(designXfull)[1], ncol = dim(designXfull)[2], byrow = T)

ind = 201
tobs = na.omit(Tobs[ind, ])
designXobs = data.frame(X1 = sin(2 * pi * tobs), X2 = sin(4 * pi * tobs), X3 = sin(6 * pi * tobs), X4 = sin(8 * pi * tobs),
                     X5 = sin(10 * pi * tobs), X6 = sin(12 * pi * tobs), X7 = sin(14 * pi * tobs), X8 = sin(16 * pi * tobs),
                     Y1 = cos(2 * pi * tobs), Y2 = cos(4 * pi * tobs), Y3 = cos(6 * pi * tobs), Y4 = cos(8 * pi * tobs),
                     Y5 = cos(10 * pi * tobs), Y6 = cos(12 * pi * tobs), Y7 = cos(14 * pi * tobs), Y8 = cos(16 * pi * tobs))
# designXobs = designXobs/matrix(c(1:8, 1:8), nrow = dim(designXobs)[1], ncol = dim(designXobs)[2], byrow = T)
FullMat = designXfull
FullMat$Y = X[ind, ]
full.fit = lm(Y ~ ., data = FullMat)

ObsMat = designXobs
ObsMat$Y = na.omit(Xobs[ind, ])
obs.fit = lm(Y ~ ., data = ObsMat)
if(sum(is.na(coef(obs.fit))) > 0){
    ObsMat = as.matrix(cbind(1, designXobs))
    coef_obs = t(ObsMat) %*% solve(ObsMat %*% t(ObsMat), na.omit(Xobs[ind, ]))
}else{
    coef_obs = coef(obs.fit)
}

plot(t, X[ind, ], type = "l", ylim = c(-6, 6), col = "green")
points(Tobs[ind, ], Xobs[ind, ], col = "blue")
lines(t, Xtrue[ind, ], col = "red")
lines(t, full.fit$fitted.values)
lines(t, as.matrix(cbind(1, designXfull)) %*% coef_obs, col = "blue")
sum(coef_obs^2)
sum(full.fit$fitted.values^2)

