import numpy as np
from scipy import sparse, io
import matplotlib.pyplot as plt
import sys
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.linalg import svds
from matplotlib.font_manager import FontProperties

def saveUserItemMatrixAsSparseMatrix(filename, R):
    R_sparse = sparse.csr_matrix(R)
    io.mmwrite(filename, R_sparse)

def loadUserItemMatrixAsSparseMatrix(filename):
    return io.mmread(filename)

def getDenseDataFromSparseMatrix(R_sparse, like = 1, dislike = -1, not_rated = 0):
    default_like = 1 
    default_dislike = -1 
    default_not_rated = 0
    R_orig = np.array(R_sparse.todense())
    R = np.zeros(R_orig.shape)
    
    R[np.where(R_orig==default_like)] = like
    R[np.where(R_orig==default_dislike)] = dislike
    R[np.where(R_orig==default_not_rated)] = not_rated
    
    I = R.copy()
    I[I != not_rated] = 1
    I[I == not_rated] = 0
    
    R_rated_indexes = R_orig.nonzero()
    
    mu = R[R_rated_indexes].mean()
    return I,R, R_rated_indexes, mu 

def get_filtered_ratings_matrix(R_train_all, R_test_all, min_num_ratings = 3):
    R_count = np.abs(R_train_all) #+np.abs(R_test_all)
    users_positions = np.where(np.sum(R_count, axis = 1)>=min_num_ratings)
    R_train_users = R_train_all[users_positions[0]]
    R_test_users = R_test_all[users_positions[0]]
    R_count_filtered = np.abs(R_train_users)
    items_positions = np.where(np.sum(R_count_filtered, axis = 0)>0)
    R_tr = R_train_users[:,items_positions[0]]
    R_ts = R_test_users[:,items_positions[0]]
    print(R_tr.shape)
    return R_tr, R_ts

def rmse(R, R_estimated, rated_set = None):
    # No es conmutativa! Primero va ground truth, normalmente R_test
    if (rated_set is None):
        nonzeros = R.nonzero()
    else:
        nonzeros = rated_set
    prediction = R_estimated[nonzeros].flatten()
    ground_truth = R[nonzeros].flatten()
    return np.sqrt(((prediction - ground_truth)**2).sum()/ground_truth.shape[0])

def accuracyWithWindow(R, R_rated_indexes, R_estimated, threshold):
    #Tripolar es una funcion que puede tomar 0, -1, 1 dependiendo del threshold
    tripolar = (abs(R_estimated[R_rated_indexes])>threshold)*(1*(R_estimated[R_rated_indexes]>threshold) - 1*(R_estimated[R_rated_indexes]<-threshold ))
    error = np.sum(((tripolar - R[R_rated_indexes])*(tripolar!=0))!=0)
    total = np.sum(tripolar!=0)
    return 1 - error/total

def accuracy(R, R_rated_indexes, R_estimated, threshold, like = 1, dislike = -1):
    total = len(R_rated_indexes[0])
    bipolar = 1*(R_estimated[R_rated_indexes]>=threshold) - 1*(R_estimated[R_rated_indexes]<threshold )
    true_positives = np.sum((R[R_rated_indexes]==like)*(bipolar>0))
    true_negatives = np.sum((R[R_rated_indexes]==dislike)*(bipolar<0))
    false_positives = np.sum((R[R_rated_indexes]==dislike)*(bipolar>0))
    false_negatives = np.sum((R[R_rated_indexes]==like)*(bipolar<0))
    acurracy = (true_positives + true_negatives)/total
    precision = true_positives/(true_positives + false_positives)
    sensitivity = true_positives/(true_positives + false_negatives) #recall
    return acurracy, precision, sensitivity, true_positives, true_negatives, false_positives, false_negatives, total

def getBaselineEstimates(R, mu, lambda1 = 0, lambda2 = 0, items_first = True, not_rated = 0):
    if not items_first:
        R = R.T
        
    (m,n) = R.shape
    bui = np.zeros((m,1))
    bii = np.zeros((n,1))

    for item in range(n):
        where = np.where(R[:,item]!=not_rated)
        item_rates = R[where,item]
        num = len(item_rates[0])
        if (num>0):
            bii[item] = (item_rates.sum() - mu*num)/(num+lambda1)
            
    for user in range(m):
        where = np.where(R[user,:]!=not_rated)
        user_rates = R[user,where]
        num = len(where[0])
        if (num>0):
            bui[user] = ((user_rates.T-bii[where]).sum() - num*mu)/(num+lambda2)
            
    if items_first:
        return bui, bii
    else:
        return bii, bui


def getMeassures(R, R_estimated, accuracy_thres = 0, like = 1, dislike = -1):
    R_rated_indexes = R.nonzero()
    acurracy, precision, sensitivity, true_positives, true_negatives, false_positives, false_negatives, total \
    = accuracy(R, R_rated_indexes, R_estimated, accuracy_thres, like, dislike)
    rmse0 = rmse(R, R_estimated)
    return acurracy, rmse0

def getStats(R_train, R_test, R_estimated, log = True, accuracy_thres = 0, like = 1, dislike = -1):
    acurracy_tr, rmse_tr = getMeassures(R_train, R_estimated, accuracy_thres, like, dislike)
    acurracy_te, rmse_te = getMeassures(R_test, R_estimated, accuracy_thres, like, dislike)
    print('accuracy (train, test): (%.4f, %.4f), rmse (train, test): (%.6f, %.6f)' %(acurracy_tr*100, acurracy_te*100, rmse_tr, rmse_te))
    return acurracy_tr, acurracy_te, rmse_tr, rmse_te

def estimate_rates(mu, bu, bi, P = np.array([0]), Q = np.array([0])):
    return mu + P.T.dot(Q) + bu + bi.T



def getRocPoint(R, R_rated_indexes, R_estimated,thres, accuracy_thres = 0, like = 1, dislike = -1):
    _, _, sensitivity, _, true_negatives, false_positives, _, _ = accuracy(R, R_rated_indexes, R_estimated,thres, accuracy_thres, like ,dislike)
    x = false_positives/(true_negatives + false_positives)
    y = sensitivity
    return x, y

def getROC(R, R_rated_indexes, R_estimated, desde = -1.5, hasta = 1.5, cantidad = 100, accuracy_thres = 0, like = 1, dislike = -1):
    x = []
    y = []
    tresholds = np.linspace(desde, hasta, cantidad)
    for thres in tresholds:
        x0, y0 = getRocPoint(R, R_rated_indexes, R_estimated,thres, accuracy_thres, like ,dislike)
        x.append(x0)
        y.append(y0)
    return x , y

def plotROC(R, R_rated_indexes, R_estimated, desde = -1.5, hasta = 1.5, cantidad = 100, thres_0 = 0, line_color = 'r', accuracy_thres = 0, like = 1, dislike = -1):
    plt.plot(*getROC(R, R_rated_indexes, R_estimated, desde, hasta, cantidad), color = line_color)
    _, _, sensitivity, _, true_negatives, false_positives, _, _ = accuracy(R, R_rated_indexes, R_estimated,thres_0, accuracy_thres, like ,dislike)
    plt.scatter(false_positives/(true_negatives + false_positives), sensitivity, color = 'g', s = 20)
    return plt

def getBaselineEstimates_SGD(R_train , R_test, R_rated_indexes_train, R_rated_indexes_test, mu,  
                           gamma= 0.01, lmbda = 0, alpha = 0, 
                           n_epochs = 100, error_calc_frec = 10,
                           accuracy_thres = 0, like = 1, dislike = -1):
    # n_epochs: Number of epochs
    # R_train: likes and dislikes user-item training matrix
    # k: Dimensionality of the latent feature space
    # lmbda: L2 regularization
    # gamma: Learning rate
    # alpha: momentum
    # sigma: P and Q standard deviation
    # error_calc_frec: calculate errors every error_calc_frec cicles
    
    m, n = R_train.shape  # Number of users and items
    
    train_rmse_vector = []
    test_rmse_vector = []
    train_accuracy_vector = [] 
    test_accuracy_vector = []
    
    #bu, bi = cf_helper.getBaselineEstimates(R_train, mu, items_first = True)
    
    bu = np.random.normal(0,0.0002,(m,1))
    bi = np.random.normal(0,0.0002,(n,1))
    
    sys.stdout.write("epoch = %s " %-1)
    acurracy_tr, acurracy_te, rmse_tr, rmse_te = getStats(R_train, R_rated_indexes_train, R_test, R_rated_indexes_test, estimate_rates(mu, bu, bi), accuracy_thres = accuracy_thres, like = like, dislike = dislike)
    
    all_users_items_cominations = list(zip(R_rated_indexes_train[0],R_rated_indexes_train[1]))
    deltaBu = 0
    deltaBi = 0
    for epoch in range(n_epochs+1):
        for u, i in all_users_items_cominations:
            error = R_train[u, i] - estimate_rates(mu, bu[u], bi[i])  # Calculate error for gradient
            
            deltaBu = gamma * ( error - lmbda * bu[u]) + alpha*deltaBu
            deltaBi = gamma * ( error - lmbda * bi[i]) + alpha*deltaBi
            bu[u] += deltaBu
            bi[i] += deltaBi
        
        if (epoch%error_calc_frec==0):
            sys.stdout.write("epoch = %s " %epoch)
            acurracy_tr, acurracy_te, rmse_tr, rmse_te = getStats(R_train, R_rated_indexes_train, R_test, R_rated_indexes_test, estimate_rates(mu, bu, bi), accuracy_thres = accuracy_thres, like = like, dislike = dislike)
            train_rmse_vector.append(rmse_tr)
            test_rmse_vector.append(rmse_te)
            train_accuracy_vector.append(acurracy_tr)
            test_accuracy_vector.append(acurracy_te)
    return train_rmse_vector, test_rmse_vector, train_accuracy_vector, test_accuracy_vector, bu, bi, gamma, lmbda 

def Matrix_Factorization_SGD(R_train , R_test, R_rated_indexes_train, R_rated_indexes_test, mu,  bu, bi, k=20 ,sigma = 0.002,
                           gamma= 0.01, lmbda = 0, alpha = 0, 
                           n_epochs = 100, error_calc_frec = 10,
                           accuracy_thres = 0, like = 1, dislike = -1):
    # n_epochs: Number of epochs
    # R_train: likes and dislikes user-item training matrix
    # k: Dimensionality of the latent feature space
    # lmbda: L2 regularization
    # gamma: Learning rate
    # alpha: momentum
    # sigma: P and Q standard deviation
    # error_calc_frec: calculate errors every error_calc_frec cicles
    
    m, n = R_train.shape  # Number of users and items
    
    P = np.random.normal(0,sigma,(k,m)) # Latent user feature matrix
    Q = np.random.normal(0,sigma,(k,n)) # Latent likes/dislikes feature matrix
    
    train_rmse_vector = []
    test_rmse_vector = []
    train_accuracy_vector = [] 
    test_accuracy_vector = []
    
    sys.stdout.write("epoch = %s " %-1)
    acurracy_tr, acurracy_te, rmse_tr, rmse_te = getStats(R_train, R_rated_indexes_train, R_test, R_rated_indexes_test, estimate_rates(mu, bu, bi, P=P, Q=Q), accuracy_thres = accuracy_thres, like = like, dislike = dislike)
    
    all_users_items_cominations = list(zip(R_rated_indexes_train[0],R_rated_indexes_train[1]))
    
    deltaP = 0
    deltaQ = 0
    deltaBu = 0
    deltaBi = 0
    for epoch in range(n_epochs+1):
        for u, i in all_users_items_cominations:
            prediction = estimate_rates(mu, bu[u], bi[i], P=P[:,u], Q=Q[:,i])
            error = R_train[u, i] - prediction  # Calculate error for gradient            
            deltaP = gamma * ( error * Q[:,i] - lmbda * P[:,u]) + alpha*deltaP
            deltaQ = gamma * ( error * P[:,u] - lmbda * Q[:,i]) + alpha*deltaQ
            
            P[:,u] += deltaP  # Update latent user feature matrix
            Q[:,i] += deltaQ  # Update latent movie feature matrix
        
        if (epoch%error_calc_frec==0):
            sys.stdout.write("epoch = %s " %epoch)
            acurracy_tr, acurracy_te, rmse_tr, rmse_te = getStats(R_train, R_rated_indexes_train, R_test, R_rated_indexes_test, estimate_rates(mu, bu, bi, P=P, Q=Q), accuracy_thres = accuracy_thres, like = like, dislike = dislike)
            train_rmse_vector.append(rmse_tr)
            test_rmse_vector.append(rmse_te)
            train_accuracy_vector.append(acurracy_tr)
            test_accuracy_vector.append(acurracy_te)
    return P, Q, train_rmse_vector, test_rmse_vector, train_accuracy_vector, test_accuracy_vector, gamma, lmbda 

def Matrix_Factorization_baselines_SGD(R_train , R_test, R_rated_indexes_train, R_rated_indexes_test, mu, k=20 ,sigma = 0.002,
                           gamma= 0.01, lmbda = 0, alpha = 0, 
                           n_epochs = 100, error_calc_frec = 10,
                           accuracy_thres = 0, like = 1, dislike = -1):
    # n_epochs: Number of epochs
    # R_train: likes and dislikes user-item training matrix
    # k: Dimensionality of the latent feature space
    # lmbda: L2 regularization
    # gamma: Learning rate
    # alpha: momentum
    # sigma: P and Q standard deviation
    # error_calc_frec: calculate errors every error_calc_frec cicles
    
    m, n = R_train.shape  # Number of users and items
    
    P = np.random.normal(0,sigma,(k,m)) # Latent user feature matrix
    Q = np.random.normal(0,sigma,(k,n)) # Latent likes/dislikes feature matrix
    
    bu = np.random.normal(0,0.0002,(m,1))
    bi = np.random.normal(0,0.0002,(n,1))
    
    train_rmse_vector = []
    test_rmse_vector = []
    train_accuracy_vector = [] 
    test_accuracy_vector = []
    
    sys.stdout.write("epoch = %s " %-1)
    acurracy_tr, acurracy_te, rmse_tr, rmse_te = getStats(R_train, R_rated_indexes_train, R_test, R_rated_indexes_test, estimate_rates(mu, bu, bi, P=P, Q=Q), accuracy_thres = accuracy_thres, like = like, dislike = dislike)
    
    all_users_items_cominations = list(zip(R_rated_indexes_train[0],R_rated_indexes_train[1]))
    
    deltaP = 0
    deltaQ = 0
    deltaBu = 0
    deltaBi = 0
    for epoch in range(n_epochs+1):
        for u, i in all_users_items_cominations:
            prediction = estimate_rates(mu, bu[u], bi[i], P=P[:,u], Q=Q[:,i])
            error = R_train[u, i] - prediction  # Calculate error for gradient            
            deltaP = gamma * ( error * Q[:,i] - lmbda * P[:,u]) + alpha*deltaP
            deltaQ = gamma * ( error * P[:,u] - lmbda * Q[:,i]) + alpha*deltaQ
            
            deltaBu = gamma * ( error - lmbda * bu[u]) + alpha*deltaBu
            deltaBi = gamma * ( error - lmbda * bi[i]) + alpha*deltaBi
            bu[u] += deltaBu
            bi[i] += deltaBi
            
            P[:,u] += deltaP  # Update latent user feature matrix
            Q[:,i] += deltaQ  # Update latent movie feature matrix
        
        if (epoch%error_calc_frec==0):
            sys.stdout.write("epoch = %s " %epoch)
            acurracy_tr, acurracy_te, rmse_tr, rmse_te = getStats(R_train, R_rated_indexes_train, R_test, R_rated_indexes_test, estimate_rates(mu, bu, bi, P=P, Q=Q), accuracy_thres = accuracy_thres, like = like, dislike = dislike)
            train_rmse_vector.append(rmse_tr)
            test_rmse_vector.append(rmse_te)
            train_accuracy_vector.append(acurracy_tr)
            test_accuracy_vector.append(acurracy_te)
    return P, Q, bu, bi, train_rmse_vector, test_rmse_vector, train_accuracy_vector, test_accuracy_vector, gamma, lmbda 

def getUsersMeans_unrated_not_count(R):
    count = (1.0*(R!=0)).sum(axis = 1).reshape(R.shape[0],1)
    means = R.sum(axis = 1).reshape(R.shape[0],1)
    count[count == 0] = 1
    means = means/count
    return means

def getUsersMeans(R):
    means = R.mean(axis = 1).reshape(R.shape[0],1)
    return means

def getItemsMeans(R):
    means = R.mean(axis = 0).reshape(R.shape[1],1)
    return means


def getPearsonSimilarityMatrix(R):
    means = getUsersMeans(R)
    R_no_dc = (R - means) 
    similarity = np.dot(R_no_dc,R_no_dc.T)
    modulus = np.sqrt((R_no_dc*R_no_dc).sum(axis = 1)).reshape(R.shape[0],1)
    denom = modulus.dot(modulus.T)
    denom[denom == 0] = 1
    similarity = similarity/denom
    return similarity

def getPearsonSimilarityMatrixMedian(R, median):
    means = median
    R_no_dc = (R - means) 
    similarity = np.dot(R_no_dc,R_no_dc.T)
    modulus = np.sqrt((R_no_dc*R_no_dc).sum(axis = 1)).reshape(R.shape[0],1)
    denom = modulus.dot(modulus.T)
    denom[denom == 0] = 1
    similarity = similarity/denom
    return similarity

def getPearsonSimilarityMatrix_non_zeros(R):
    means = getUsersMeans_unrated_not_count(R)
    R_no_dc = (R - means)*(R!=0) 
    similarity = np.dot(R_no_dc,R_no_dc.T)
    modulus = np.sqrt((R_no_dc*R_no_dc).sum(axis = 1)).reshape(R.shape[0],1)
    denom = modulus.dot(modulus.T)
    denom[denom == 0] = 1
    similarity = similarity/denom
    return similarity

def getJaccardSimilarityMatrix(R):
    R_abs = 1.0*(abs(R)>0)
    intersect = R_abs.dot(R_abs.T)
    users_count = R_abs.sum(axis = 1)
    users_count = users_count.reshape(users_count.shape[0],1)
    denom = users_count + users_count.T
    denom = denom - intersect
    denom[denom==0] = 1
    similarity = np.dot(R,R.T)
    similarity = similarity/denom
    return similarity

def getCosineSimilarityMatrix(R):
    similarity = np.dot(R,R.T)
    modulus = np.sqrt((R*R).sum(axis = 1)).reshape(R.shape[0],1)
    denom = modulus.dot(modulus.T)
    denom[denom == 0] = 1
    similarity = similarity/denom
    return similarity


def calcultaSimilarityMatrix(R, dist_type = 'cosine', max_co_rated = 0,save = False):
    if dist_type == 'jaccard_with_negatives':
        similarityMatrix = getJaccardSimilarityMatrix(R)
    elif dist_type == 'pearson':
        similarityMatrix = getPearsonSimilarityMatrix(R)
    else:
        distanceMatrix = pairwise_distances(R, metric=dist_type)
        similarityMatrix = 1 - distanceMatrix
    if max_co_rated>0:
        R_abs = abs(R)
        prod = R_abs.dot(R_abs.T)/max_co_rated
        #np.fill_diagonal(prod,0)
        prod[prod>1] = 1
        
        #prod = np.ones(similarityMatrix.shape)
        similarityMatrix = similarityMatrix*prod
    if save:
        np.save('similarityMatrix_'+dist_type,similarityMatrix)
    return similarityMatrix

def predictions(R, similarityMatOrig, divide_by_weights_sum = True, count_diag = False, means = 0):
    # divide_by_weights_sum -> Divide por la suma de los pesos y no por la cantidad de elementos likeados/dislikeados
    similarityMat = similarityMatOrig.copy()
    if not count_diag:
        np.fill_diagonal(similarityMat,0)
    difMat = (R-means).T.dot(similarityMat).T
    if divide_by_weights_sum:
        denomin = abs(similarityMat)[:,::-1].sum(axis = 1)
    else: 
        denomin = abs(R.T).sum(axis=1)
    denomin[denomin == 0] = 1
    nomalizer = abs(R.T).sum(axis=1)
    nomalizer[nomalizer == 0] = 1
    if divide_by_weights_sum:
        result = (difMat.T/denomin).T
    else:
        result = difMat/denomin
    result = result + means
    return result

def predictions_normalized(R, similarityMatOrig, divide_by_weights_sum = True, count_diag = False, means = 0):
    # divide_by_weights_sum -> Divide por la suma de los pesos y no por la cantidad de elementos likeados/dislikeados
    user_std = R.std(axis = 1).reshape(R.shape[0],1)
    user_std[user_std==0] = 1
    similarityMat = similarityMatOrig.copy()
    if not count_diag:
        np.fill_diagonal(similarityMat,0)
    difMat = ((R-means)/user_std).T.dot(similarityMat).T
    if divide_by_weights_sum:
        denomin = abs(similarityMat)[:,::-1].sum(axis = 1)
    else: 
        denomin = abs(R.T).sum(axis=1)
    denomin[denomin == 0] = 1
    nomalizer = abs(R.T).sum(axis=1)
    nomalizer[nomalizer == 0] = 1
    if divide_by_weights_sum:
        result = (difMat.T/denomin).T
    else:
        result = difMat/denomin
    result = user_std*result + means
    return result

def predictions_K_neighbours(users_items_matrix, similarityMatOrig, k = 10, means = 0, divide_by_weights_sum = True, count_diag = False):
    similarityMat = similarityMatOrig.copy()
    if not count_diag:
        np.fill_diagonal(similarityMat,0)
    
    sim_index_sorted_not_trunkated = np.argsort(abs(similarityMat), axis = 1)[:,::-1]
    
    for u in range(similarityMat.shape[0]):
        similarityMat[u, sim_index_sorted_not_trunkated[u,k:]] = 0

    predictMatrix = users_items_matrix.T.dot(similarityMat).T
    
    if divide_by_weights_sum:
        denomin = abs(similarityMat).sum(axis = 1)
    else:
        #Esta mal esto, habria que cambiarlo y tener en cuanto solo la cantidad que participa de la suma
        denomin = abs(users_items_matrix.T).sum(axis=1)
    
    denomin[denomin == 0] = 1
    
    
    if divide_by_weights_sum:
        result = (predictMatrix.T/denomin).T
    else:
        result = predictMatrix/denomin
    result = result + means
    return result


def SVD(users_items_train_matrix, k = 20):
    #get SVD components from train matrix. Choose k.
    u, s, vt = svds(users_items_train_matrix, k)
    s_diag_matrix=np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    return X_pred


def get_relevant_item_position(R_est, user, item, number_of_random_items = 1000):
    random_items = np.random.choice(R_est.shape[1], size=number_of_random_items, replace=False)
    if (item not in random_items):
        random_items = np.insert(random_items,np.random.randint(number_of_random_items), item)
    
    where = np.where(random_items==item)[0][0]
    item_rates = np.argsort(R_est[user, random_items])[::-1]
    return 100*np.where(item_rates == where)[0][0]/(random_items.shape[0]-1)

def get_percentiles(R, R_est, like = 1, number_of_random_items = 1000, points = 100):
    relevant_indexes = np.where(R == like)
    relevant_indexes = zip(relevant_indexes[0],relevant_indexes[1])
    ratings = []
    for user, item in relevant_indexes:
        position = get_relevant_item_position(R_est, user, item, number_of_random_items = number_of_random_items)
        ratings.append(position)
        
    x = 100*np.array(range(points+1))/points
    return np.percentile(np.array(ratings), x),x

def get_personalization_index(R_est, top_k = 10, searching_in = 10):
    sorted_index_mat = np.argsort(R_est , axis = 1)[:,::-1]
    n_user = sorted_index_mat.shape[0]
    suma = 0
    count = 0
    for user0 in range(n_user):
        for user in range(user0+1, n_user):
            aux = np.intersect1d(sorted_index_mat[user0,:searching_in], sorted_index_mat[user,:top_k], assume_unique = True)
            suma = suma + len(aux)
            count = count + 1
    return 1-suma/(count*top_k)

def get_item_avg_ratings(R):
    # Quiza no esta bien esto, analizar
    normalizer = np.abs(R).sum(axis = 0)
    normalizer[normalizer == 0] = 1
    avg_items_ratings = R.sum(axis = 0)/normalizer
    return avg_items_ratings

def get_popularity_index(R, R_est, top_k = 10, searching_in = 10):
    sorted_index_mat = np.argsort(R_est , axis = 1)[:,::-1]
    avg_items_ratings = get_item_avg_ratings(R)
    users_dif_sorted_indexes = np.argsort(avg_items_ratings)[::-1][:top_k]
    n_user = sorted_index_mat.shape[0]
    #print(sorted_index_mat[0,:10])
    suma = 0
    count = 0
    for user0 in range(n_user):
        aux = np.intersect1d(sorted_index_mat[user0,:searching_in], users_dif_sorted_indexes, assume_unique = True)
        suma = suma + len(aux)
        #if (len(aux)>2):
        #    print(user0)
        count = count + 1
    return suma/(count*top_k)

def getUsersStats(R):
    # Get R stats without counting zeros
    count = (1.0*(R!=0)).sum(axis = 1).reshape(R.shape[0],1)
    means = R.sum(axis = 1).reshape(R.shape[0],1)
    count[count == 0] = 1
    means = means/count
    R_no_dc = (R - means)*(R!=0)
    sigmaSqr = (R_no_dc*R_no_dc).sum(axis = 1).reshape(R.shape[0],1)
    desv = np.sqrt(sigmaSqr)/count
    means[means == 0] = means[means!=0].mean()
    return means, desv


def plot_percentiles(options_vector, rmse_array, perc_array, percentiles_train_media = None, parameter_name='parameter_name',title= 'Titulo', featured_idx = -1, featured_text = '', xlim=[0,2], ylim=[0,35]):
    f, ([ax1, ax2]) = plt.subplots(1,2, sharex=False, sharey=False, figsize=(12, 6))
    colors = ['b','g','y','m','r','b','k','y','g']
    plt_legends = []
    fontP = FontProperties()
    fontP.set_size('small')


    for i in range(len(options_vector)):
        if i==featured_idx:
            label = '%s %s, rmse = %.4f'%(parameter_name,featured_text,rmse_array[i])
            marker = '.'
        else:
            label = '%s %s, rmse = %.4f'%(parameter_name,options_vector[i],rmse_array[i])
            marker = None
        plt_legend, = ax1.plot(*perc_array[i],  marker=marker, color = colors[i], label=label)
        ax2.plot(*perc_array[i], marker=marker, color = colors[i], label = label)
        plt_legends.append(plt_legend)
    if percentiles_train_media:
        ax1.plot(*percentiles_train_media, linestyle = ":", color = 'k')

    ax1.legend(handles=plt_legends, loc = 'best', prop = fontP)
    ax2.legend(handles=plt_legends, loc = 'best', prop = fontP)

    ax1.set_xlabel('K [%], default = 1000')
    ax1.set_ylabel('hits [%]')

    ax2.set_xlabel('K [%], default = 1000')
    ax2.set_ylabel('hits [%]')


    ax2.set_ylim(ylim)
    ax2.set_xlim(xlim)

    f.suptitle(title, fontsize = 15)
    f.show()


class Testing:
    def get_testing_rates(m,n, not_rated = 0):
        # Generate a small matrix to test stuff. It has zeros and one row of zeros 
        R = np.random.normal(0,2,(m,n))
        R[abs(R)<1] = not_rated
        R[1,:] = not_rated
        return R

    def get_pearson_similarity(Rt):
        m,n = Rt.shape
        w = np.zeros((m,m))
        for a in range(m):
            for u in range(m):
                means_a = Rt[a].mean()
                means_u = Rt[u].mean()
                Rta = (Rt[a] - means_a)
                Rtu = (Rt[u] - means_u)
                for i in range(n):
                    w[a,u] =  w[a,u] + (Rta[i]*Rtu[i])

                desv_a = np.sqrt((Rta**2).sum())
                desv_u = np.sqrt((Rtu**2).sum())
                deno = desv_a*desv_u
                if deno!=0:
                    w[a,u]= w[a,u]/(deno)
                else:
                    w[a,u]=0   
        return w

    def get_pearson_similarity_non_zeros(Rt, not_rated = 0):
        #Zeros mean not rated
        m,n = Rt.shape
        w = np.zeros((m,m))
        for a in range(m):
            for u in range(m):
                nz_a = Rt[a]!=not_rated 
                nz_u = Rt[u]!=not_rated 
                c_a = (nz_a!=not_rated).sum()
                c_u = (nz_u!=not_rated).sum()
                if c_a == not_rated:
                    c_a = 1
                if c_u == not_rated:
                    c_u = 1   
                means_a = Rt[a, nz_a].sum()/c_a
                means_u = Rt[u, nz_u].sum()/c_u
                Rta_nz = (Rt[a] - means_a)*(Rt[a]!=not_rated)
                Rtu_nz = (Rt[u] - means_u)*(Rt[u]!=not_rated)
                for i in range(n):
                    if ((Rt[a,i]!=not_rated)&(Rt[u,i]!=not_rated)):
                        w[a,u] =  w[a,u] + (Rta_nz[i]*Rtu_nz[i])

                desv_a = np.sqrt((Rta_nz[Rta_nz.nonzero()]**2).sum())
                desv_u = np.sqrt((Rtu_nz[Rtu_nz.nonzero()]**2).sum())
                deno = desv_a*desv_u
                if deno!=0:
                    w[a,u]= w[a,u]/(deno)
                else:
                    w[a,u]=0   
        return w