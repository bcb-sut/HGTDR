from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from matplotlib import pyplot

def AUROC(scores, labels):
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    ns_probs = [0 for _ in range(len(labels))]
    lr_auc = roc_auc_score(labels, scores)
    write_to_out('AUROC: %.3f \n' % (lr_auc))
    ns_fpr, ns_tpr, _ = roc_curve(labels, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(labels, scores)
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, label='Logistic')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.savefig('../out/AUROC', dpi=180)
    pyplot.show()

def AUPR(scores, labels):
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    lr_precision, lr_recall, _ = precision_recall_curve(labels, scores)
    lr_auc = auc(lr_recall, lr_precision)
    write_to_out('AUPR: %.3f \n' % (lr_auc))
    no_skill = len(labels[labels==1]) / len(labels)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, label='HGT')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.savefig('../out/AUPR', dpi=180)
    pyplot.show()
    
def plot_losses(losses, val_losses):
    pyplot.plot(range(len(losses)), losses, label="loss")
    pyplot.plot(range(len(losses)), val_losses, label="val_loss")
    pyplot.legend()
    pyplot.savefig('../out/losses', dpi=200)
    pyplot.clf()
    
def write_to_out(text):
    print(text)
    out_file = open('../out/out.txt', 'a')
    out_file.write(text)
    out_file.close()