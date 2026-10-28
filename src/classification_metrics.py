# Precision and F1 from confusion matrix
def precision_f1(tp, fp, fn):
	# no alerts means precision is undefined
	if tp + fp == 0:
		precision = None
	else:
		precision = round(100 * tp / (tp + fp), 2)
	# F1 from counts avoids dividing by zero
	denom = 2 * tp + fp + fn
	f1 = round(2 * tp / denom, 4) if denom else 0.0
	return precision, f1