import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
import matplotlib.pyplot as plt
import seaborn as sns

# P(C|D) = P(D|C) * P(C)
# P(D|C) = Product(P(x1|C))

def byesian_probability(data_frame, feature_set, target, class_name):
	overall = len(data_frame)
	target_class = data_frame[lambda df: df[target] == class_name]
	p_class = len(target_class) / overall
	result = p_class
	for (key, value) in feature_set.items():
		p_feature = len(target_class[lambda df: df[key] == value]) / len(target_class) + 0.0001
		result *= p_feature
	return result

def predict_dict(data_frame, values_set, target_column):
	target_column_values = np.unique(data_frame[target_column])
	results = {}
	for val in target_column_values:
		results[val] = (byesian_probability(df, values_set, target_column, val))
	import operator
	max_p = max(results.items(), key=operator.itemgetter(1))[0]
	confidence = results[max_p] / sum(results.values())
	return max_p, confidence

def predict(data_frame, values_data_frame, target_column):
	result_data_frame = pd.DataFrame(columns=[target_column, 'Confidence'])

	list_of_dicts = values_data_frame.to_dict('records')
	results = []
	for values_to_predict in list_of_dicts:
		p_class, p_confidence = predict_dict(data_frame, values_to_predict, target_column)
		results.append([p_class, p_confidence])
	return pd.DataFrame(np.array(results), columns=[target_column, 'Confidence'])

if __name__ == '__main__':
	import argparse
	import sys

	parser = argparse.ArgumentParser(description='Naive Bayesian classifier.')
	parser.add_argument('target', nargs='?', help='Target label column to predict.')
	parser.add_argument('train', nargs='?', help='Train CSV file. Labeled features.')
	parser.add_argument('predict', nargs='?', help='Predict CSV file. Non-labeled features.')
	parser.add_argument('--output', nargs='?', help='Output predicted CSV file.')

	args = parser.parse_args(sys.argv[1:])
	if not args.predict and not args.train:
		parser.print_help()
		exit(0)

	df = pd.read_csv(args.train)
	df.interpolate(method='pad', inplace=True)
	df.dropna(inplace=True)

	df_predict = pd.read_csv(args.predict)
	df_result = predict(df, df_predict, args.target)

	if args.output:
		df_result.to_csv(args.output)
	else:
		print(df_result)
