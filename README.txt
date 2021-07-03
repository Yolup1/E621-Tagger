This is a classification Neural Network for Identifying image Tags of the Furry Porn Variety.

you will need :
Tensorflow >= 2.4
Pandas >= 1.2.3
matplotlib >= 3.34
numpy > = 1.19.2
and a feeling of emptiness inside >= 1.0.0

the model is saved as only weights, this means it needs to be built at runtime, however it also means you get to see the code that makes the network arcitecture. this will allow you to load and modify the model after the fact, for other projects you may have in mind.
this model can input any sized image and you can even disable the auto resizing to 512-512 (it keeps aspect ratio) however, this is not advised, as the network was trained on 128x 256x and 512x scales.
it is currently set to glob png files in the directory it is run from, so just pop those bad bois in there and you're ready to rock. if you have jpgs and want to use those find the line of code that has a ('*.png') and change the .png to .jpg and thats it!