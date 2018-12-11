from sklearn.metrics import log_loss, roc_auc_score
import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    print("*"*150)

    num_labels = train_y.shape[1]
    patience = 5


    ###################### VALIDATION LOSS #####################################
    best_loss = -1
    best_weights = None
    best_epoch = 0
    
    current_epoch = 0
    
    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_loss = 0
        for j in range(num_labels):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= num_labels

        print("Epoch {0} loss {1} best_loss {2}".format(current_epoch, total_loss, best_loss))

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == patience:
                break

    model.set_weights(best_weights)

    # for i in range(3):
    #     model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
    return model


    ######################## ROC AUC SCORE #####################################
    # best_score = -1
    # best_weights = None
    # best_epoch = 0
    
    # current_epoch = 0

    # while True:
    #     model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
    #     y_pred = model.predict(val_x, batch_size=batch_size)

    #     total_score = 0
    #     for j in range(num_labels):
    #         try:
    #             score = roc_auc_score(val_y[:, j], y_pred[:, j])
    #         except:
    #             score = 0
    #         total_score += score

    #     total_score /= num_labels

    #     print("Epoch {0} score {1} best_score {2}".format(current_epoch, total_score, best_score))

    #     current_epoch += 1
    #     if total_score > best_score or best_score == -1:
    #         best_score = total_score
    #         best_weights = model.get_weights()
    #         best_epoch = current_epoch
    #     else:
    #         if current_epoch - best_epoch == patience:
    #             break

    # model.set_weights(best_weights)
    # return model  

    ############################## KERAS DeFAULT ###############################
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # bst_model_path = 'model_weights' + '.h5'
    # model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, 
    #                                 save_weights_only=True)

    # hist = model.fit(train_x, train_y,
    #                 validation_data=(val_x, val_y),
    #                 epochs=50, batch_size=batch_size, shuffle=True,
    #                 callbacks=[early_stopping, model_checkpoint])
            
    # model.load_weights(bst_model_path)
    # bst_val_score = min(hist.history['val_loss'])
    # print(bst_val_score)

    # return model



def train_folds(X, y, X_test, fold_count, batch_size, get_model_func):
    fold_size = len(X) // fold_count
    models = []
    result_path = "/home/zafar/Desktop/predictions"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for fold_id in range(0, fold_count):
        print("fold_id")
        print(fold_id)
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = np.array(X[fold_start:fold_end])
        val_y = np.array(y[fold_start:fold_end])

        model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y)
        train_predicts_path = os.path.join(result_path, "train_predicts{0}.npy".format(fold_id))
        test_predicts_path = os.path.join(result_path, "test_predicts{0}.npy".format(fold_id))
        train_predicts = model.predict(X, batch_size=512, verbose=1)
        test_predicts = model.predict(X_test, batch_size=512, verbose=1)
        np.save(train_predicts_path, train_predicts)
        np.save(test_predicts_path, test_predicts)
        # models.append(model)

    return models
