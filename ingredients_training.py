import ingredients_utils as ing_utils

import numpy as np

MODEL_NAME = "ingredients"

dm = ing_utils.DataModule(
    dir_path="ingredients/"
)

x_train, x_valid, y_label_train, y_label_valid, y_loc_train, y_loc_valid, x_data_max, y_loc_max = dm.get_data(
    valid_size=0.2,
    # is_canny_used=True,
    # canny_threshold1=192,
    # canny_threshold2=192
)
print(np.shape(x_train), np.shape(y_label_train), np.shape(y_loc_train))
print(np.shape(x_valid), np.shape(y_label_valid), np.shape(y_loc_valid))

tm = ing_utils.TrainModule(
    input_shape=np.shape(x_train[0, :, :, :]),
    ckpt_path=f"ckpt/{MODEL_NAME}/{MODEL_NAME}.ckpt",
    model_path=f"model/{MODEL_NAME}.h5",
    log_dir=f"log/{MODEL_NAME}/"
)

model = tm.build_model()
model.summary()
train_hist, model = tm.train_model(model,
                                   x_train=x_train, y_label_train=y_label_train, y_loc_train=y_loc_train,
                                   x_valid=x_valid, y_label_valid=y_label_valid, y_loc_valid=y_loc_valid)

print(x_data_max, y_loc_max)
