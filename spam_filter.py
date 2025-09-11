from scipy.io import loadmat

MAT_PATH = "emails.mat" 

def load_spam_mat(path=MAT_PATH):
    data = loadmat(path)
    print(data.keys())