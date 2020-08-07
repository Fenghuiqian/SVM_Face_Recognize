#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import cvxopt # smo solver
import cv2


# In[2]:


def make_ovr_label(y):
    label = []
    for i in list(set(y)):
        temp = []
        for l in y:
            if l == i:
                temp.append(1.0)
            else:
                temp.append(-1.0)
        label.append(temp)
    return label


# ##### Linear SVM,   img resize to (112/4,  92/4) 

# In[36]:


img_h = int(112/4)
img_w = int(92/4)

# 5 fold validation
for t in [[1,2],[3,4], [5,6], [7,8], [9,10]]:
    accuracy = []
    test = t
    train = [i for i in range(1,11) if i not in test]
    train_x = np.zeros((40*8 ,img_h*img_w))
    train_y = []
    cnt = 0
    for i in train:
        for p in range(1,41):
            img_name = "%s_%s.png" % (p, i)
            img = cv2.imread("ATT/%s" % img_name, 0).astype(np.float64)
            img = cv2.resize(img, (img_w ,img_h))
            train_x[cnt,:] = img.reshape(-1)
            cnt += 1
            train_y.append(p)
#     train_y = np.array(train_y).reshape(1,-1)
    test_x = np.zeros((40*2 ,img_h*img_w))
    test_y = []
    cnt = 0
    for i in test:
        for p in range(1,41):
            img_name = "%s_%s.png" % (p, i)
            img = cv2.imread("ATT/%s" % img_name, 0).astype(np.float64)
            img = cv2.resize(img, (img_w ,img_h))
            test_x[cnt, :] = img.reshape(-1)
            cnt += 1
            test_y.append(p)
    test_y = np.array(test_y)
    
    # ----------------------------------
    # G
    # add bias column
#     X = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis=1)
    test_x = np.concatenate((test_x, np.ones((test_x.shape[0], 1))), axis=1)
    
    # h
    label = make_ovr_label(train_y)
    test_y = make_ovr_label(test_y)
    
    # 
    
    for i, l in enumerate(label):
#         h = cvxopt.matrix(np.array(l))
        X = np.concatenate((train_x, np.ones((train_x.shape[0], 1))), axis=1)
        for se, val in enumerate(l):
            if val ==1.0:
                X[se, :] = X[se, :] * (-1)
        G = cvxopt.matrix(X)
        h = cvxopt.matrix(np.array([-1.0]*len(l)))
            # P
        iden = np.identity(X.shape[1])
        iden[iden.shape[1]-1][iden.shape[1]-1] = 0
        P = cvxopt.matrix(iden)
        # q
        q = cvxopt.matrix(np.zeros([X.shape[1]]))
        
        # solver
        sol = cvxopt.solvers.qp(P, q, G, h)
        # so, got the weights
        w = np.array(sol['x'])
        # prediction
        test_predict = test_x @ w
        # calculate accuracy
        test_y_label = test_y[i]
        correct = 0
        for c, pred in enumerate(test_predict.reshape((80,))):
            if pred < 0 and test_y_label[c] < 0:
                correct +=1
            elif pred > 0 and test_y_label[c] > 0:
                correct +=1
    # print(correct/test_predict.shape[0])
        accuracy.append(correct/(test_predict.shape[0]))


    # plot accuracy
    plt.bar(range(len(accuracy)), accuracy)
    plt.xlabel('classes')
    plt.ylim(0.98,1)
    plt.ylabel('accuracy')
    plt.title('accuracy of 5 predictions')
    plt.show()


# ##### polynomial_kernel SVM

# In[3]:


def polynomial_kernel(x, z, poly=2):
    return (1 + np.dot(x, z)) ** poly


# In[4]:


epsilon = 0


# In[32]:


img_h = int(112/4)
img_w = int(92/4)

# 5 fold validation
for t in [[1,2],[3,4], [5,6], [7,8], [9,10]]:
    accuracy = []
    test = t
    train = [i for i in range(1,11) if i not in test]
    train_x = np.zeros((40*8 ,img_h*img_w))
    train_y = []
    cnt = 0
    for i in train:
        for p in range(1,41):
            img_name = "%s_%s.png" % (p, i)
            img = cv2.imread("ATT/%s" % img_name, 0).astype(np.float64)
            img = cv2.resize(img, (img_w ,img_h))
            train_x[cnt,:] = img.reshape(-1)
            cnt += 1
            train_y.append(p)
#     train_y = np.array(train_y).reshape(1,-1)
    test_x = np.zeros((40*2 ,img_h*img_w))
    test_y = []
    cnt = 0
    for i in test:
        for p in range(1,41):
            img_name = "%s_%s.png" % (p, i)
            img = cv2.imread("ATT/%s" % img_name, 0).astype(np.float64)
            img = cv2.resize(img, (img_w ,img_h))
            test_x[cnt, :] = img.reshape(-1)
            cnt += 1
            test_y.append(p)
    test_y = np.array(test_y)
    
    
    label = make_ovr_label(train_y)
    test_y = make_ovr_label(test_y)
    
    
    # ----------------------------------
    
    for i, l in enumerate(label):
        
        Y_train = np.array(l)
        length = train_x.shape[0]

        y_gram = np.outer(Y_train, Y_train)

        x_gram = np.zeros((length, length))
        for u in range(length):
            for j in range(length):
                x_gram[u][j] += polynomial_kernel(train_x[u], train_x[j])

        # SOLVER
        P = cvxopt.matrix(y_gram * x_gram)
        q = cvxopt.matrix(np.ones(length) * -1)  # 'q' must be a 'd' matrix with one column

        A = cvxopt.matrix(Y_train, (1, length))
        b = cvxopt.matrix(0.0) 

        G = cvxopt.matrix(np.identity(length) * -1)
        h = cvxopt.matrix(np.zeros(length))
        # 
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(sol['x'])
        index = [ii for ii in range(len(alpha)) if alpha[ii] > epsilon]

        # Support Vector
        support_vector = []
        for ind in index:
            support_vector.append(train_x[ind])
        support_vector = np.array(support_vector)

        # BIAS
        b = 0
        for j in index:
            sigma = 0
            for le in range(length):
                sigma += alpha[le] * l[le] * x_gram[le, j]
            b += l[j] - sigma
        b = b/len(index)

        # predict x 
        sigma = np.zeros(len(test_x))
        for pos in range(len(test_x)):
            for alp in range(len(alpha)):
                sigma[pos] += alpha[alp] * l[alp] * polynomial_kernel(test_x[pos], train_x[alp])
        test_y_label = test_y[i]
        
        # judge
        correct = 0
        for pos, sign in enumerate(np.sign(sigma)):
            if np.sign(np.array(test_y_label))[pos] == sign:
                correct += 1
        accuracy.append(correct/len(test_y_label))
        
    # plot accuracy
    plt.bar(range(len(accuracy)), accuracy)
    plt.xlabel('classes')
    plt.ylim(0,1)
    plt.ylabel('accuracy')
    plt.title('accuracy of 5 predictions')
    plt.show()

    


# In[ ]:





# In[ ]:





# In[ ]:




