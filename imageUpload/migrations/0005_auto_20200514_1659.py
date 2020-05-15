# Generated by Django 3.0.6 on 2020-05-14 11:29

from django.db import migrations, models
import imageUpload.models


class Migration(migrations.Migration):

    dependencies = [
        ('imageUpload', '0004_auto_20200514_1646'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageuploadmodel',
            name='image1',
            field=models.ImageField(default='', upload_to=imageUpload.models.product_image_upload_path_1),
        ),
        migrations.AlterField(
            model_name='imageuploadmodel',
            name='image2',
            field=models.ImageField(default='', upload_to=imageUpload.models.product_image_upload_path_2),
        ),
    ]
