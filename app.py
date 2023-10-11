import os
import re
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from ctc_layer import CTCLayer
import requests
from PIL import Image


model_path = 'model_tf==2.9.0.h5'
img_width = 177
img_height = 40
max_length = 5
characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'e', 'p']


model = keras.models.load_model(model_path, custom_objects={'CTCLayer': CTCLayer})
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)


char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def encode_single_prediction(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    return img


def decode_single_prediction(pred):
    input_len = np.array([pred.shape[1]])
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = tf.strings.reduce_join(num_to_char(results[0])).numpy().decode("utf-8")
    return output_text


def predict(temp_file_path):
    image = encode_single_prediction(temp_file_path)
    prediction = prediction_model.predict(image)
    result = decode_single_prediction(prediction)
    return result


def solve_captcha(input_string):
    pattern = r'(\d+)p(\d+)(e)?$'
    match = re.match(pattern, input_string)
    if match:
        num1 = int(match.group(1))
        num2 = int(match.group(2))
    return num1 + num2


def fetch_and_convert_image():
    url = "https://obs.iste.edu.tr/oibs/captcha/CaptchaImg.aspx"
    response = requests.get(url)
    if response.status_code == 200:
        temp_aspx_path = "temp_captcha.aspx"
        with open(temp_aspx_path, "wb") as f:
            f.write(response.content)

        # Convert the ASPX image to PNG
        aspx_image = Image.open(temp_aspx_path)
        png_image_path = "temp_captcha.png"
        aspx_image.save(png_image_path, format="PNG")
        os.remove(temp_aspx_path)
        return png_image_path
    else:
        raise Exception("Failed to fetch the image")


def main():
    st.title("Proliz OBS system CAPTCHA solver")

    if st.button("Generate New CAPTCHA and solve"):
        try:
            fetched_image_path = fetch_and_convert_image()
            with Image.open(fetched_image_path) as img:
                st.image(img, caption="Fetched CAPTCHA", use_column_width=True)
            if fetched_image_path is not None:
                prediction = predict(fetched_image_path)
                result = solve_captcha(prediction)
                st.markdown(f"<h2>Answer is: {result}</h2>", unsafe_allow_html=True)
                os.remove(fetched_image_path)
        except Exception as e:
            st.error(f"Error fetching CAPTCHA: {str(e)}")
            return


if __name__ == '__main__':
    main()