Project Proposal\
Members: Wang Ting Shu,\
Due date: March 4, 2022\
Course: CSC413

Face Mask Detection \
Introduction\
It has been 2 years since COVID-19 entered into our lives. Most of us have gotten used to washing our hands often, wearing a mask, physical distancing, and other things to protect ourselves from the virus. A lot of places depend on people wearing masks to protect the people with weaker immune systems like hospitals and retirement homes, or to prevent a mass spread of the virus like in food factories. These facilities won’t always have people to ensure people wears mask. So, we want to train an algorithm that can detect if a person is wearing a mask, not wearing a mask, or wearing a mask incorrectly. And using this model we will apply it to videos or security cameras that is able to detect face masks in real time.

Related works\
	In a paper written by Qin and Li (1), they have combined using image super-resolution with SRC-Net to detect the wearing condition of face mask of a person. The algorithm reached a 98.70% accuracy. The images were first processed by cropping the image to the bounding box of the face detected, then super-resolution was applied, trained the resulting image with SRC-Net then predicting.
	In the above paper, they have used SRC-Net to train the model, while Buele, S.Talahua, Calvonpiña and Varela-Aldás (2) using MobileNetV2 reached a accuracy of 99.65%. We can observe that just by using CNN, we are able to achieve a high accuracy of detecting if someone is wearing a face mask.
	On another hand, we can approach this problem using visual transformers(ViT), which can beat some state-of-the-art CNN by four times fewer computing resources. (3) But the publication of EfficientNet V2 has proven to be having  even better results than ViT by training 5 – 11 times faster than ViT with the same computer resources.


Method / Algorithm\
	Our goal is to train our dataset against two different computer vision algorithms and compare the results. Using the data set that we have obtained from Kaggle, we will first preprocess the images by cropping out the regions of interest. Since the accuracy of our model depends on the quality of our images, we will apply image super-resolution and resize the images into a fixed size. Using the new data that we have processed, we will train 80% of our data, with the other 10% as validation set and 10% as our test set. We will train our data against two different methods in computer vison. One will be trained using a pretrained EffcientNet V2, and the other will be the combination of ViT and ResNet-50 as the backbone. Then by fine tuning the hyperparameters of the algorithms, we can get a model that detects whether someone is wearing a mask not. We would then apply this model to videos so it can be tested that it can run in real time.

Citation\
[1] Qin, B., & Li, D. (2020). Identifying facemask-wearing condition using image super-resolution with classification network to prevent COVID-19. Sensors, 20(18), 5236.

[2] Talahua, J. S., Buele, J., Calvopiña, P., &amp; Varela-Aldás, J. (2021, June 18). Facial recognition system for people with and without face mask in times of the COVID-19 pandemic. MDPI. Retrieved March 4, 2022, from https://www.mdpi.com/2071-1050/13/12/6900/htm

[3]Boesch, G. (2022, January 17). Vision transformers (VIT) in Image Recognition - 2022 guide. viso.ai. Retrieved March 4, 2022, from https://viso.ai/deep-learning/vision-transformer-vit/

[4] Ibrahim, M. (2021, April 3). Google releases EfficientNetV2 - a smaller, faster, and better EfficientNet. Medium. Retrieved March 5, 2022, from https://towardsdatascience.com/google-releases-efficientnetv2-a-smaller-faster-and-better-efficientnet-673a77bdd43c





