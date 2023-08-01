from django.db import models

# Create your models here.
class student(models.Model):
    name=models.CharField(max_length=20,null=True,blank=True)
    sid=models.IntegerField()
    date=models.DateTimeField(auto_now=True)
    def __str__(self):
        return self.name


