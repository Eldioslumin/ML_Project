from scipy.io import loadmat
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


MAT_PATH = "emails.mat" 
SEED = 21092000

def load_spam_mat(path=MAT_PATH):
    data = loadmat(path)
    X = data["X"]
    Y = data["Y"]
    
    X = X.T  
    Xcsr = X.tocsr()
    # We transpose our original data and then tranform the Coordinate sparse matrix 
    # into a Compressed Sparse Row, which is easier to handle.
    
    y = np.ravel(Y)       
    y = (y == 1).astype(int) 
    # We flatten the data and transform the 1/-1 into 1/0 because of problems with sklearn otherwise.
    
    return Xcsr, y

def create_holdout_split(X, y, test_size=0.2, random_state=SEED):
    # Due to the high number of instances, 
    # Training/Validation/Test split should work.
    #  
    # The importance of splitting our data into training, validation, and test sets
    # steams from our need to evaluate the performance of our model
    # in unseen data. 
    # In this case, since our original data came in the form of a premade bag of words
    # based on a corpus we do not have access to, doing the split becomes more relevant,
    # since we cannot use external data sources to increase our number of training instances.
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=random_state, stratify=y)
    # stratify preserves the ratio of of spam/non-spam in both the training and the test sets.
    
    # Then we split training data for validation for hyperparameters.
    # This leaves us with a ~70% of instances for training, ~20% for validation, and 10 % for testing.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.222, random_state=42, stratify=y_train)
    

    
    
    return X_train, X_test, X_validation, y_train, y_test, y_validation



def find_optimal_threshold(model, X_validation, y_validation, limit_fp_rate=0.002):

    y_probability = model.predict_proba(X_validation)[:, 1]
    fp_rate, recall, thresholds = roc_curve(y_validation, y_probability)

    mask = fp_rate <= limit_fp_rate
    if not np.any(mask):
        print("No threshold found for this model. :/")
        return 0.5 
    # So if no threshold actually complies with our requirements, we simply let it be.

    fp_under = fp_rate[mask]
    recall_valid = recall[mask]
    thresholds_valid = thresholds[mask]

    minimal_fp = np.min(fp_under)
    index = np.where(fp_under == minimal_fp)[0]
    best = index[np.argmax(recall_valid[index])]
    # First we find our fp rate that actually complies with the company's limit.
    # Then we check which limit also comes with the best recall possible
    # since it is a kind of secondary mission for us.

    return thresholds_valid[best]



def evaluate_model(model, X_test, y_test, threshold=0.5):
    
    y_probability = model.predict_proba(X_test)[:, 1]
    y_prediction = (y_probability >= threshold).astype(int)
    

    tn, fp, fn, tp = confusion_matrix(y_test, y_prediction).ravel()
    # We calculate all common metrics.
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc = roc_auc_score(y_test, y_probability)
    
    return accuracy, precision, recall, fp_rate, confusion_matrix(y_test, y_prediction), auc
    

def plot_results(accuracy, precision, recall, auc, confusion_matrix, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    cm = confusion_matrix
    sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title(f'{model_name} Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['Non-Spam', 'Spam'])
    ax1.set_yticklabels(['Non-Spam', 'Spam'])
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    values = [accuracy, precision, recall, auc]
    
    bars = ax2.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax2.set_title(f'{model_name} Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    filename = f"{model_name.replace(' ', '_').lower()}_results.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    
    X, y = load_spam_mat()

    X_train, X_test, X_validation, y_train, y_test, y_validation = create_holdout_split(X, y)
    
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    # "The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document 
    # is to scale down the impact of tokens that occur very frequently in a given corpus 
    # and that are hence empirically less informative 
    # than features that occur in a small fraction of the training corpus."
    
    #By setting smooth_idf to True, we assume a fake additional document that has all terms in it. 
    # This prevents 0 divisions.
    
    X_train = tfidf.fit_transform(X_train)
    X_validation = tfidf.transform(X_validation)
    X_test = tfidf.transform(X_test)

    # We apply tf-idf. 
    # tf(t,d) × idf(t,D)
    # tf(t, d) = f(t, d) / sum(f(t', d) for all terms t' in d)
    # idf(t, D) = log( N / (1 + df(t)) )
    # First we create the data split

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'SVM': SVC(probability=True, random_state=SEED, C=1.0),  # probability=True for predict_proba
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    }
    # Then we select all models we will use.
    # The advantage of using all models from sklearn is that they all use similar functions to get results/evaluate
    # so my life is easier.
    
    best = { 
    "best_recall" : 0,
    "best_fp_rate" : 1,
    "best_model" : None,
    }
    
    for name, model in models.items(): 
          
        model.fit(X_train, y_train)
        threshold = find_optimal_threshold(model, X_validation, y_validation)
        
        accuracy, precision, recall, fp_rate, confusion_matrix, auc = evaluate_model(model, X_test, y_test, threshold)
        
        # 1) We train our models.
        # 2) Find the best threshold for them.
        # 3) Evaluate them.
        
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"FP Rate: {fp_rate:.4f} ({fp_rate*100:.2f}%)")
        print(f"AUC: {auc:.3f}")
        
        plot_results(accuracy, precision, recall, auc, confusion_matrix, name)
        
        if fp_rate <= 0.002:
            print("The model meets the 0.2% FP limit")
            if fp_rate < best['best_fp_rate']:
                best['best_model'] = name
                best['best_fp_rate'] = fp_rate
                best['best_recall'] = recall
            elif fp_rate == best['best_fp_rate']:
                if best['best_recall'] < recall:
                    best['best_model'] = name
                    best['best_fp_rate'] = fp_rate
                    best['best_recall'] = recall
        else:
            print("The model does not meet the 0.2% FP limit")
        
        
    if best['best_model'] is None:
            print("No model was found to meet the company's requirements. :-(")        
                
                
    print(f"\nThe best model during test is {best['best_model']}")
    print(f"Spam detected: {best['best_recall']:.1%}")
    print(f"FP rate: ({best['best_fp_rate']*100:.2f}%)")


if __name__ == "__main__":
    main()