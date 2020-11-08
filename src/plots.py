import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix)
import numpy as np
import itertools as tools

def plot_aurc_plot(Y_true, Y_pred, title="ROC Curve"):
    
    fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(fpr, tpr,lw=2)
    ax.plot([0,1],[0,1],c='violet',ls='--')
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])

    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title(title,fontsize=16)
    custom_lines = [Line2D([0], [0], color='blue', lw=4)]
    ax.legend(custom_lines, ['AUC = {:.2f}'.format(roc_auc_score(Y_true,Y_pred))], loc=(.77,.115))

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    ax.yaxis.set_label_coords(-0.1,0.5)
    ax.xaxis.set_label_coords(0.5,-0.1)
    print(f"\033[1m\033[94mROC AUC score = ", roc_auc_score(Y_true, Y_pred))

    plt.savefig('../graphs/{0}_ROC_Curve.png'.format(title), dpi=300)

# def plot_confusion_matrix(results, class_names):
# 	def plot_matrix(cm, classes, title ='Confusion Matrix', cmap=plt.cm.Blues):

# 		cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]
# 		plt.imshow(cm, interpolation='nearest', cmap = cmap)
# 		plt.title(title)
# 		plt.colorbar()
# 		tick_marks = np.arange(len(classes))
# 		plt.xticks(tick_marks, classes, rotation = 45)
# 		plt.yticks(tick_marks, classes)

# 		formatType = '.2f'
# 		maxLimit = cm.max() / 2.
# 		for x, y in tools.product(range(cm.shape[0]), range(cm.shape[1])):
# 			plt.text(y, x, format(cm[x, y], formatType), horizontalalignment = "center", color = "white" if cm[x, y] > maxLimit else "black")
# 		plt.ylabel('True Label')
# 		plt.xlabel('Predicted Label')
# 		plt.tight_layout()
		
# 	yLabelTrue, yLabelPred = zip(* results)

# 	cnf_matrix_internal = confusion_matrix(yLabelTrue, yLabelPred)
# 	np.set_printoptions(precision = 2)

# 	#Plot Confusion Matrix
# 	plt.figure()
# 	plot_matrix(cnf_matrix_internal, classes = class_names, title ='Normalized Confusion Matrix')
# 	plt.savefig('../graphs/{0}_Confusion_Matrix.png'.format(title), dpi=300)
