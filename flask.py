import flask
import werkzeug, os
import tensorflow.keras.models
import numpy
import skimage.io

IMGS_FOLDER = os.path.join('static', 'imgs')

app_url = "http://192.168.43.177:5000" #"https://hiai.website/vae_mnist" 

app = flask.app.Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMGS_FOLDER

encoder = tensorflow.keras.models.load_model("VAE_encoder.h5")
decoder = tensorflow.keras.models.load_model("VAE_decoder.h5")

im_id = 0

@app.route("/", methods=["POST", "GET"])
def vae():
    subject = flask.request.args.get("subject")
    print(subject)

    if subject == "encode":
        return upload_encode_image(flask.request)
    elif subject == "decode":
        return decode_img(flask.request)
    else:
        return flask.redirect(flask.url_for("static", filename="main.html"))

def upload_encode_image(encode_request):
    if "imageToUpload" not in encode_request.files:
        return "<html><body><h1>No file uploaded.</h1><a href=" + app_url + ">Try Again</a></body></html>"
    img = encode_request.files["imageToUpload"]
    if img.filename == '':
        return "<html><body><h1>No file uploaded.</h1><a href=" + app_url + ">Try Again</a></body></html>"
    filename = werkzeug.utils.secure_filename(img.filename)
    _, file_ext = filename.split(".")
    if file_ext.lower() not in ["jpg", "jpeg", "png"]:
        return "<html><body><h1>Wrong file extension. The supported extensions are JPG, JPEG, and PNG.</h1><a href=" + app_url + ">Try Again</a></body></html>"

    read_image = skimage.io.imread(fname=filename, as_gray=True)
    if read_image.shape[0] != 28 or read_image.shape[1] != 28:
        return "<html><body><h1>Image size must be 28x28 ...</h1><a href=" + app_url + ">Try Again</a></body></html>"        

    return encode_img(read_image)

def encode_img(img):
    test_sample = numpy.zeros(shape=(1, 28, 28, 1))
    test_sample[0, :, :, 0] = img
    test_sample = test_sample.astype("float32") / 255.0

    latent_vector = encoder.predict(test_sample)
    return flask.render_template("encode_result.html", num1=latent_vector[0, 0], num2 = latent_vector[0, 1])

def decode_img(decode_request):
    global im_id
    num1, num2 = decode_request.form["num1"], decode_request.form["num2"]
    
    latent_vector  = numpy.zeros(shape=(1, 2))
    latent_vector[0, 0] = num1
    latent_vector[0, 1] = num2
    print(latent_vector)
    
    decoded_image = decoder.predict(latent_vector)
    decoded_image = decoded_image[0, :, :, 0]
    
    saved_im_name = os.path.join(app.config['UPLOAD_FOLDER'], "vae_result_" + str(im_id) + ".jpg")
    im_id = im_id + 1
    skimage.io.imsave(fname=saved_im_name, arr=decoded_image)

    return flask.render_template("decode_result.html", img_name=saved_im_name)

app.run(host="192.168.43.177", port=5000, debug=True)
