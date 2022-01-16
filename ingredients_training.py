import ingredients_utils as ing_utils

import numpy as np

dm = ing_utils.DataModule(
    dir_path="ingredients/"
)

x_train, x_valid, y_label_train, y_label_valid, y_loc_train, y_loc_valid, x_data_max, y_loc_max = dm.get_data(
    valid_size=0.2
)
print(np.shape(x_train), np.shape(y_label_train), np.shape(y_loc_train))
print(np.shape(x_valid), np.shape(y_label_valid), np.shape(y_loc_valid))

tm = ing_utils.TrainModule(
    input_shape=np.shape(x_train[0, :, :, :]),
    ckpt_path="ckpt/ingredients.ckpt",
    model_path="ingredients.h5",
    log_dir="log/"
)

model = tm.build_model()
model.summary()
train_hist, model = tm.train_model(model,
                                   x_train=x_train, y_label_train=y_label_train, y_loc_train=y_loc_train,
                                   x_valid=x_valid, y_label_valid=y_label_valid, y_loc_valid=y_loc_valid)

print(x_data_max, y_loc_max)
