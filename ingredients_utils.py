from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from natsort import natsorted
import xml.etree.ElementTree as ElemTree
import numpy as np
import cv2
import os
import re

from tensorflow.keras import models, layers, activations, initializers, optimizers, losses, callbacks, metrics
import tensorflow as tf


class DataModule:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        if self.dir_path[-1] != "/":
            raise Warning("dir_path must end with a '/'")
        if os.path.isdir(self.dir_path) is not True:
            raise NotADirectoryError(f"{dir_path} is not a directory.")

    def get_data(self, valid_size=None, norm=True, is_canny_used=False, canny_threshold1=127, canny_threshold2=127):
        x_data_max, y_loc_max = 255.0, 227.0

        def canny_edge(raw_img, threshold1: int, threshold2: int):
            canny_output = cv2.Canny(raw_img, threshold1, threshold2)
            return canny_output

        p = re.compile("[.][x][m][l]")
        xml_file_list = list()
        for file_name in os.listdir(f"{self.dir_path}"):
            if p.search(file_name):
                xml_file_list.append(file_name)
        xml_file_list = natsorted(xml_file_list)

        x_data = list()
        y_loc = list()
        y_label = list()
        for xml_file_name in xml_file_list:
            tree = ElemTree.parse(f"{self.dir_path}{xml_file_name}")
            root = tree.getroot()

            file_name = os.path.splitext(xml_file_name)[0]
            img = cv2.imread(f"{self.dir_path}{file_name}.jpg")
            if is_canny_used:
                img = canny_edge(img, threshold1=canny_threshold1, threshold2=canny_threshold2)
            if norm:
                img = img / x_data_max
            x_data.append(img)

            loc = list()
            for data in root[6][4]:
                loc.append(int(data.text))
            y_loc.append(loc)
            y_label.append(root[6][0].text)

        x_data, y_label, y_loc = np.array(x_data), np.array(y_label), np.array(y_loc)
        if is_canny_used:
            x_data = np.expand_dims(x_data, axis=-1)
        le = LabelEncoder()
        y_label = to_categorical(le.fit_transform(y_label))
        y_loc = np.asarray(y_loc, dtype=np.float) / y_loc_max

        if valid_size:
            x_train, x_valid, y_label_train, y_label_valid, y_loc_train, y_loc_valid = train_test_split(x_data,
                                                                                                        y_label,
                                                                                                        y_loc,
                                                                                                        test_size=valid_size)
            return x_train, x_valid, y_label_train, y_label_valid, y_loc_train, y_loc_valid, x_data_max, y_loc_max

        return x_data, y_label, y_loc, x_data_max, y_loc_max


class IoU(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(IoU, self).__init__(**kwargs)

        self.iou = self.add_weight(name="iou", initializer="zeros")
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
        self.num_ex = self.add_weight(name="num_ex", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        def get_loc(y):
            y = y * 227
            return y[:, 0], y[:, 1], y[:, 2], y[:, 3]

        def get_area(x1, y1, x2, y2):
            return tf.math.abs(x2 - x1) * tf.math.abs(y2 - y1)

        gt_x1, gt_y1, gt_x2, gt_y2 = get_loc(y_true)
        p_x1, p_y1, p_x2, p_y2 = get_loc(y_pred)

        i_x1 = tf.maximum(gt_x1, p_x1)
        i_y1 = tf.maximum(gt_y1, p_y1)
        i_x2 = tf.minimum(gt_x2, p_x2)
        i_y2 = tf.minimum(gt_y2, p_y2)

        i_area = get_area(i_x1, i_y1, i_x2, i_y2)
        u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(p_x1, p_y1, p_x2, p_y2) - i_area

        iou = tf.math.divide(i_area, u_area)
        self.num_ex.assign_add(1)
        if tf.reduce_mean(iou).numpy() > 2:
            self.total_iou.assign_add(tf.constant(1e-9))
        else:
            self.total_iou.assign_add(tf.reduce_mean(iou))
        self.iou = tf.math.abs(tf.math.divide(self.total_iou, self.num_ex))

    def result(self):
        return self.iou

    def reset_state(self):
        self.iou = self.add_weight(name="iou", initializer="zeros")
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
        self.num_ex = self.add_weight(name="num_ex", initializer="zeros")


class TrainModule:
    def __init__(self, input_shape, ckpt_path: str, model_path: str, log_dir: str):
        self.input_shape = input_shape
        self.ckpt_path = ckpt_path
        if os.path.exists(os.path.dirname(self.ckpt_path)) is False:
            os.makedirs(os.path.dirname(self.ckpt_path))
        self.model_path = model_path
        self.log_dir = log_dir
        if self.log_dir[-1] != "/":
            raise Warning("dir_path must end with a '/'")
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)

    def build_model(self):
        input_layer = layers.Input(shape=self.input_shape, name="image")

        x = input_layer
        for i in range(0, 6):
            conv_filters = 2 ** (4 + i)
            x = layers.Conv2D(conv_filters, (3, 3), padding="same", activation=activations.relu,
                              kernel_initializer=initializers.he_uniform())(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPool2D((2, 2), padding="same")(x)

        x = layers.Flatten()(x)

        x = layers.Dense(512, activation=activations.relu,
                         kernel_initializer=initializers.he_uniform())(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(128, activation=activations.relu,
                         kernel_initializer=initializers.he_uniform())(x)

        class_out = layers.Dense(3, activation=activations.softmax,
                                 kernel_initializer=initializers.he_uniform(), name="class_out")(x)
        box_out = layers.Dense(4, name="box_out")(x)

        model = models.Model(input_layer, [class_out, box_out])
        model.compile(
            optimizer=optimizers.Adam(),
            metrics={
                "class_out": metrics.categorical_accuracy,
                "box_out": IoU(name="iou")
            },
            loss={
                "class_out": losses.categorical_crossentropy,
                "box_out": losses.MSE
            },
            run_eagerly=True
        )

        return model

    def train_model(self, model: models,
                    x_train: np.ndarray, y_label_train: np.ndarray, y_loc_train: np.ndarray,
                    x_valid: np.ndarray, y_label_valid: np.ndarray, y_loc_valid: np.ndarray):
        hist = model.fit(
            x={"image": x_train}, y={"class_out": y_label_train, "box_out": y_loc_train},
            batch_size=16,
            epochs=1000,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_box_out_iou", patience=31, verbose=1, mode="max"),
                callbacks.ModelCheckpoint(filepath=self.ckpt_path,
                                          monitor="val_box_out_iou",
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=True,
                                          mode="max"),
                callbacks.ReduceLROnPlateau(monitor="val_box_out_iou",
                                            factor=0.5,
                                            patience=6,
                                            verbose=1,
                                            mode="max",
                                            min_lr=1e-8),
                callbacks.TensorBoard(log_dir=self.log_dir),
            ],
            validation_data=({"image": x_valid}, {"class_out": y_label_valid, "box_out": y_loc_valid}),
            steps_per_epoch=300
        )
        model.load_weights(filepath=self.ckpt_path)
        model.save(filepath=self.model_path)

        return hist, model
