#create rda file 

#dataset structure
save(heart, file = "rda_files/structure.rda")

#optimal values
save(opt_p, opt_p_two, opt_cp, opt_mtry, opt_cost, file = "rda_files/opt_values.rda")

save(model_glm, model_glm_two, model_nb, model_tree, model_rf, model_svm, model_nb_final, model_tree_final, model_rf_final, file = "rda_files/models.rda" )

#confusion matricies
save(cm_glm, cm_glm_two, cm_nb, cm_tree, cm_rf, cm_svm, cm_ens, cm_ens_final, file = "rda_files/matrices.rda")

#results table

save(results, file = "rda_files/results.rda")

