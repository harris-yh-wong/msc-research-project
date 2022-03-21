#! check Python import system
# https://docs.python.org/3/reference/import.html

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np


# Plot the distribution of variable
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%  ({v:d})'.format(p=pct, v=val)
    return my_autopct


# Make bin plot for tendency
def bin_plot(dataset, var_name, tgt, bins):
    # sort the value
    new_name = var_name + "_bin"
    dataset.sort_values(by=var_name)
    # can use both qcut & cut function to make bin
    dataset[new_name] = pd.cut(
        dataset[var_name], bins, precision=0, right=False)
    groups = dataset[tgt].groupby(dataset[new_name])
    gp_default = groups.sum() / groups.count()
    gp_frame = gp_default.to_frame()
    gp_frame['odds'] = list(map(logodds, gp_frame[tgt].values.tolist()))
    print(gp_frame)
    # fit the line
    gp_frame.dropna(inplace=True)
    length = range(1, len(gp_frame)+1)
    fig, ax = plt.subplots()
    fit = np.polyfit(length, gp_frame['odds'], deg=1)
    # make the plot
    ax.plot(length, fit[0] * length + fit[1], color='red')
    ax.scatter(x=length, y=gp_frame['odds'])
    plt.xlabel('Group of %s' % (var_name))
    plt.ylabel('Chances of  PHQ >= 10')
    plt.show()
    # drop group name
    dataset.drop([new_name], axis=1, inplace=True)
    return


def splitDataSet(dataset, split_size=5):
    num_line = 0
    onefile = dataset.groupby(by=["id"]).size().index.tolist()
    num_line = len(onefile)
    arr = np.arange(num_line)  # get a seq and set len=numLine
    np.random.shuffle(arr)  # generate a random seq from arr
    list_all = arr.tolist()
    each_size = int(num_line / split_size)  # size of each split sets
    print('The dataset is split into', split_size, 'groups.')
    print('Each group have', each_size, 'ids.')
    split_all = []
    each_split = []
    count_num = 0
    count_split = 0
    for i in range(len(list_all)):
        each_split.append(onefile[int(list_all[i])].strip())
        count_num += 1
        if count_num == each_size:
            count_split += 1
            array_ = np.array(each_split)
            split_all.append(each_split)
            each_split = []
            count_num = 0
    return split_all


def generateDataset(X_dataset, y_dataset, id_list):
    X_train_all = []
    X_valid_all = []
    y_train_all = []
    y_valid_all = []
    for i in range(len(id_list)):
        model_split = copy.deepcopy(id_list)
        vallid_id = model_split.pop(i)
        train_id = np.array(model_split).flatten().tolist()
        X_train_all.append(
            X_dataset[X_dataset['id'].isin(train_id)].drop(['id'], axis=1))
        X_valid_all.append(
            X_dataset[X_dataset['id'].isin(vallid_id)].drop(['id'], axis=1))
        y_train_all.append(
            y_dataset[y_dataset['id'].isin(train_id)].drop(['id'], axis=1))
        y_valid_all.append(
            y_dataset[y_dataset['id'].isin(vallid_id)].drop(['id'], axis=1))
    return X_train_all, X_valid_all, y_train_all, y_valid_all


def crossValidation(clf, X_train_all, X_valid_all, y_train_all, y_valid_all):
    Recalls = []
    F1s = []
    for i in range(len(X_train_all)):
        train_X = X_train_all[i]
        train_y = y_train_all[i].target
        valid_X = X_valid_all[i]
        valid_y = y_valid_all[i].target

        Recall, F1 = classifier(clf, train_X, train_y, valid_X, valid_y)
        Recalls.append(Recall)
        F1s.append(F1)

        Recall_mean = sum(Recalls)/len(Recalls)
        F1s_mean = sum(F1s)/len(F1s)

        return Recall_mean, F1s_mean


def classifier(clf, train_X, train_y, test_X, test_y):
    # train with randomForest
    clf = clf.fit(train_X, train_y)
    # test Classifier with valid sets
    predict_ = clf.predict(test_X)
    proba = clf.predict_proba(test_X)
    score_ = clf.score(test_X, test_y)
    # Modeal Evaluation
    recall = recall_score(test_y, predict_, average='weighted')
    f1 = f1_score(test_y, predict_, average='weighted')
    return recall, f1


# Evaluate the model by AUC
fpr, tpr, threshold = roc_curve(y_test.target, y_pred)
roc_auc = auc(fpr, tpr)
# make the ROC plot
plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_curve')
plt.legend(loc="lower right")
plt.show()


# Plot feature importance
forest_importances = pd.Series(classifier_rf_bal.feature_importances_,
                               index=X_train.columns.values).sort_values(ascending=False)[:10]
fig, ax = plt.subplots()
forest_importances.sort_values().plot.barh()
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.show()


# Reshape the dataset for Neural Network
def reshape_feature(input_data, feature_list, sort_value):
    dataset = input_data.loc[:, feature_list].sort_values(sort_value)
    sequence = []
    uni = []
    id = dataset.values[0][0]
    for i in dataset.values:
        values = i[2:]
        if i[0] == id:
            sequence.append(list(values))
        else:
            id = i[0]
            uni.append(sequence)
            sequence = []
            sequence.append(list(values))
    return uni


# Reshape the target for Neural Network
def reshape_target(input_data):
    Y = input_data.loc[:, ["id_new",
                           "phq_score"]].sort_values("id_new").groupby(['id_new']).head(1)
    Y["target"] = Y["phq_score"] >= 10
    Y = Y.reset_index(drop=True)
    return Y["target"].astype(int)


# Set the length of sequence
def pad_sequences(sequence, max_len, len_num):
    for i in range(len(sequence)):
        if len(sequence[i]) < max_len:
            sequence[i].end([[0] * len_num] * (max_len - len(sequence[i])))
        else:
            sequence[i] = sequence[i][:max_len]
    return sequence


# Trans 1D dimension to 2D dimension Type1: 3 * (30 * 12)
def dimension_trans_1(input_data):
    final_data = []
    for id in range(len(input_data)):
        feature_dim = []
        data_trans = np.transpose(input_data[id]).tolist()
        for i in data_trans:
            new_m = []
            for j in range(12):
                new_m.append(i[15*j:15*(j+2)])
            feature_dim.append(new_m)
        data = moveaxis(feature_dim, 0, 2)
        final_data.append(data)
    return final_data


# Trans 1D dimension to 2D dimension Type2: 3 * (105 * 105)
def dimension_trans_2(input_data):
    final_data = []
    for id in range(len(input_data)):
        feature_dim = []
        data_trans = np.transpose(input_data[id]).tolist()
        for i in data_trans:
            new_m = []
            for j in range(105):
                new_m.append(i[j:j+105])
            feature_dim.append(new_m)
        data = moveaxis(feature_dim, 0, 2)
        final_data.append(data)
    return final_data


# Plot learning curve of Neural Network
def plot_learningCurve(history, epoch):
    # Plot training & validation accuracy values
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


# Plot training & validation loss values
plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# LSTM model structure
def build_LSTM_model():
    model = Sequential()
    model.add(LSTM(16, input_shape=(210, 3)))
    # LSTM model structure
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


# CNN model structure 1
def CNN_model1():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(2, 2),
              padding='valid', input_shape=(12, 30, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


# CNN model structure 2
def CNN_model2():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(2, 2),
              padding='valid', input_shape=(105, 105, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, kernel_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


# Build the Neural Network model
print('Build model...')
model_1 = build_LSTM_model()
print('=====================================================================')
print('Compile model...')
model_1.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
print('=====================================================================')
print('Train...')
history = model_1.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      shuffle=True,
                      verbose=2,
                      validation_split=0.5)
print('Train Finished!')
