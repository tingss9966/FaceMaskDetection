# FaceMaskDetection
This is a project for CSC413.\
It has been 2 years since COVID-19 entered into our lives. Most of us have gotten used to washing our hands often, wearing a mask, physical distancing, and other things to protect ourselves from the virus. A lot of places depend on people wearing masks to protect the people with weaker immune systems like hospitals and retirement homes, or to prevent a mass spread of the virus like in food factories. These facilities won’t always have people to ensure people wears mask. So, we want to train an algorithm that can detect if a person is wearing a mask, not wearing a mask, or wearing a mask incorrectly. And using this model we will apply it to videos or security cameras that is able to detect face masks in real time.\

1. LoadData: Preprocess the data and save and the process images to a .npz file.
2. WithBackBone: run two models with backbone to compare accuracy and run time
3. cnn: convolution network with 8 million parameters face detection classifier
4. videoFD: run to see application of this classifier. Note: Need to run at least WithBackbone or cnn first)