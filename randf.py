from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rf(train_x, train_y, test_x, test_y):
    score_l = []
    for i in range(0, 100, 1):
        rfc = RandomForestClassifier(n_estimators=i+1, random_state=90)                     
        rfc = rfc.fit(train_x, train_y)                    
        s = rfc.score(test_x, test_y)
        score_l.append(s)
    score_max = max(score_l)  
    print("The max score: {}".format(score_max), 
            "N_estimators: {}".format(score_max * 1 + 1))

    x = np.arange(1, 101, 1)
    plt.subplot(111)
    plt.plot(x, score_l, 'r-')
    plt.show()
