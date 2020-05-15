from django.db import models
import os


def path_and_rename_1(path):
    def wrapper(instance, filename):
        ext = 'jpg'        
        # set filename as random string
        filename = '{}.{}'.format('Image1', ext)
        # return the whole path to the file
        return os.path.join(path, filename)
    return wrapper
def path_and_rename_2(path):
    def wrapper(instance, filename):
        ext = 'jpg'        
        # set filename as random string
        filename = '{}.{}'.format('Image2', ext)
        # return the whole path to the file
        return os.path.join(path, filename)
    return wrapper

product_image_upload_path_1 = path_and_rename_1('images/')
# assign it `__qualname__`
product_image_upload_path_1.__qualname__ = 'product_image_upload_path_1'

product_image_upload_path_2 = path_and_rename_2('images/')
# assign it `__qualname__`
product_image_upload_path_2.__qualname__ = 'product_image_upload_path_2'

class ImageUploadModel(models.Model):
    image1 = models.ImageField(upload_to=product_image_upload_path_1, default='', height_field=None, width_field=None, max_length=100)
    image2 = models.ImageField(upload_to=product_image_upload_path_2, default='' , height_field=None, width_field=None, max_length=100)
    
    def __str__(self):
        return 'imageUpload'