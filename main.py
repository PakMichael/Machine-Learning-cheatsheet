from PretrainedAdaptable import VGG16
from utils import *

model = VGG16()

img=getImage('D:/1.jpg')
img = transform(img).unsqueeze(0)

res=model(img)
score,index=res[0].max(1)

print(getLabels()[str(index.item())])