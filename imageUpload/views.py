from django.shortcuts import render
from .models import ImageUploadModel
from .forms import ImageUploadForm
import os
import time
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PhotoCombine.settings import MEDIA_ROOT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

content_path = os.path.join(MEDIA_ROOT,'images/Image1.jpg')
style_path = os.path.join(MEDIA_ROOT,'images/Image2.jpg')

def get_image( img_path, max_resolution):
        max_dimension = max_resolution
        img = Image.open(img_path)
        long = max(img.size)
        scale = max_dimension / long
        # Resizing the image and converting it to RGB so all image types are usable
        img = img.convert("RGB")
        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
        img = img_to_array(img)
        # We need to broadcast the image array such that it has a batch dimension
        img = np.expand_dims(img, axis=0)
        return img

def preprocess_image(path):
        
  return tf.keras.applications.vgg19.preprocess_input(path)

def deprocess_image(processed_img):
     
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  x[:, :, 0] += vgg_mean[2]
  x[:, :, 1] += vgg_mean[1]
  x[:, :, 2] += vgg_mean[0]
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x


def create_vgg_model():
    vgg_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg_model.trainable = False
    # We don't need to (or want to) train any layers of our pre-trained vgg model, so we set it's trainable to false.
    # Get output layers corresponding to style and content layers
    style_outputs = [vgg_model.get_layer(name).output for name in style_layers]
            
    content_outputs = [vgg_model.get_layer(name).output for name in content_layers]
            
    output_layers = style_outputs + content_outputs
    # Build model
    return models.Model(vgg_model.input, output_layers)

 
def content_loss(content, target):
  return tf.reduce_mean(tf.square(content - target))
  
def gram_matrix( input_tensor):
  num_channels = int(input_tensor.shape[-1])
  # if input tensor is a 3D array of size Nh x Nw X Nc
  # we reshape it to a 2D array of Nc x (Nh*Nw)
  input_vectors = tf.reshape(input_tensor, [-1, num_channels])
  num_vectors = tf.shape(input_vectors)[0]
  gram = tf.matmul(input_vectors, input_vectors, transpose_a=True)
  return gram / tf.cast(num_vectors, tf.float32)

#derived from refrences
def style_loss(style, gram_target):

        
  gram_style = gram_matrix(style)

        
  return tf.reduce_mean(tf.square(gram_style - gram_target))

def get_feature_representations():
        

  content_image = processed_content_image
        
  style_image = processed_style_image 
        
  # batch compute content and style features
  style_outputs = model(style_image)
  content_outputs = model(content_image)

   # Get the style and content feature representations from our model
  style_features = [style_layer[0]
                          for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0]
                            for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features

def compute_loss(loss_weights, init_image, gram_style_features,
                      content_features):
  
  style_weight, content_weight = loss_weights
  # Feed our init image through our model. This will give us the content and
  # style representations at our desired layers.
  model_outputs = model(init_image)

  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]

  total_style_score = 0
  total_content_score = 0
 
  averge_style_weight = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    
     total_style_score += averge_style_weight * style_loss(comb_style[0], target_style)

  # content losses from all layers
  average_content_weight = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features,
                                                content_output_features):
    total_content_score += average_content_weight * content_loss(comb_content[0], target_content)
        
  total_style_score *= style_weight
  total_content_score *= content_weight

  # Get total loss
  total_loss = total_style_score + total_content_score 
  
  return total_loss, total_style_score, total_content_score

def compute_gradients(config):
  with tf.GradientTape() as tape:
    all_loss = compute_loss(**config)
        # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, config['init_image']), all_loss

def run_style_transfer( num_iterations,
                           content_weight,
                           style_weight               
                       ):
  
  
  for layer in model.layers:
    layer.trainable = False

  # Get the style and content feature representations
  style_features, content_features = get_feature_representations()
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

  # Initially set content image as our base output image
  init_image = processed_content_image
  init_image = tf.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  # Here Adam Optimizer is used to optimize the loss, but in paper LBFGS is recommended.

  opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
  

  # Store our best result
  best_loss, best_img = float('inf'), None

  
  loss_weights = (style_weight, content_weight)
 
  config = {
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features,
        }

        
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means

  imgs = []


  for i in range(num_iterations):
    
    grads, all_loss = compute_gradients(config)
    loss, _, _ = all_loss
    opt.apply_gradients([(grads, init_image)])
    # Clip image to be in range 0-255 
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    if i%10 == 0:             
    #printing every 10th Iteration and loss
      print("Iteration ",i)
      print("LOSS= {0}".format(loss))
    if loss < best_loss:
    # Update best loss and best image from total loss.
      best_loss = loss
      best_img = deprocess_image(init_image.numpy())
    if i % 100 == 0:
      imgs.append(deprocess_image((init_image.numpy())))

  plt.imshow(best_img)
  
  return best_img, best_loss


def plot():
  plt.figure(figsize=(30,30))
  plt.subplot(5,5,1)
  plt.title("Content Image",fontsize=20)
  img_cont = load_img(content_path)
  plt.imshow(img_cont)

  plt.subplot(5,5,1+1)
  plt.title("Style Image",fontsize=20)
  img_style = load_img(style_path)
  plt.imshow(img_style)

  plt.subplot(5,5,1+2)
  plt.title("Final Image",fontsize=20)
  plt.imshow(a)
  plt.savefig('static/final.jpg',bbox_inches='tight')


def finalRun():
    global content_img_arr
    global style_img_arr
    global content_img_arr
    global processed_content_image
    global processed_style_image
    global vgg_mean
    global content_layers
    global style_layers
    global num_content_layers
    global num_style_layers
    global model
    global a
    global b
    content_img_arr = get_image(content_path,512)

    style_img_arr = get_image(style_path,512)

    content_img_arr.shape

    processed_content_image = preprocess_image(content_img_arr)

    processed_style_image = preprocess_image(style_img_arr)

    vgg_mean = [123.68, 116.779, 103.939]

    content_layers = ['block4_conv2']
            # Style layer we are interested in
    style_layers = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',
                ]
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    model = create_vgg_model()

    a,b = run_style_transfer(num_iterations=10,
                           content_weight=1e-1,
                           style_weight=1e2,
                           )


def uploadView(request):
    pass

def index(request):
    if request.method == 'POST': 
        form = ImageUploadForm(request.POST, request.FILES) 
        if form.is_valid():
            saved_data = form.save()                        
            try:                
                print('Images GEtting Deleted From Database')
                remove_image = ImageUploadModel.objects.get(id = saved_data.pk)
                finalRun()
                plot()
                if remove_image.image1 and remove_image.image2:                  
                  remove_image.image1.delete()
                  remove_image.image2.delete()
                remove_image.delete()
            except ImageUploadModel.DoesNotExist:
                print('Model Query Doesnt Exist')
    else: 
        form = ImageUploadForm() 
    return render(request, 'index.html', {'form' : form})