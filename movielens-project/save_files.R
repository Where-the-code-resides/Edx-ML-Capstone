#create rda file 


#optimal values
save(opt_lambda, opt_lambda_two, file = "rda_files/opt_values.rda")

save(mu,
     mean,
     preds_movie_bias,
     preds_movie_user_bias,
     preds_movie_bias_reg,
     preds_movie_user_bias_reg,
     preds_movie_user_bias_reg_final,
     file = "rda_files/predictions.rda" )

#rmse results
save(rmse_results, file = "rda_files/rmse_results.rda")

#rmse values
save(rmse_mean,
     rmse_movie_bias,
     rmse_movie_user_bias,
     rmse_movie_bias_reg,
     rmse_movie_user_bias_reg,
     rmse_movie_user_bias_reg_final,
     file = "rda_files/rmse_values.rda")