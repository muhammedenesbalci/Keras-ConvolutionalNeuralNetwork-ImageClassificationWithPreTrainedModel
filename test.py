from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet import preprocess_input

full_model = load_model('C:\\Users\\enes\\Desktop\\githubb\\readymodel\\model.h5')
full_model.summary()

# Predictions
def predict_(img_path):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return full_model.predict(x)


result = predict_("C:\\Users\\enes\\Desktop\\githubb\\readymodel\\dataset\\single_images\\img.jpg")
print(result)

if (result[0][0] == 1):
    print("Cat!")
elif (result[0][1] == 1):
    print("Dog!")
else:
    print("something else!")
