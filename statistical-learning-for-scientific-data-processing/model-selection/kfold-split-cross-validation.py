from sklearn.model_selection import KFold, cross_val_score

X = ["a", "a", "b", "c", "c", "c"]

k_fold = KFold(n_splits=3)

print("Indicies of Training and Test Data for Each Cross Validation Fold:")
print("------------------------------------------------------------------")
for train_indicies, test_indicies in k_fold.split(X):
	print('Train: %s | test: %s' % (train_indicies, test_indicies))
print()

[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) 
	for train, test in k_fold.split(X_digits)]

