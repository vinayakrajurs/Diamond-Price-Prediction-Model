# Diamond-Price-Prediction-Model

# Motivation:
The main purpose of the Expert System is to replicate knowledge and skills of human experts in a particular area, and then to use this knowledge to solve problems without human intervention. The aim of this project was to implement all the knowledge gained from this workshop into an actual working model. The idea behind choosing  the diamond price predictor model was that the dataset consisted of float values and was relatively easy to understand and implement. 

# Procedure:
1. We started off by searching for datasets on kaggle and other sources on the net. 
2. We had initially thought of using a stock price predicting dataset as it had all of its features consisting of float data type.
3. Since the model became a bit too complicated, we decided to shift to something more simpler and that's when we chose to go ahead with the diamonds price prediction model.
4. The dataset was simple and contained almost no junk values.
5. We came up with the initial draft of the model using Keras Sequential Model and tensorflow and realised that the features required were Carat, Length, Width and Thickness.
6. This Model had some complications and wasn’t perfect as it had a very high MAE (mean absolute error) value.
7. Through implementing feature crosses, we were able to bring down the MAE value significantly.
8. Once the model was trained, we loaded the different weights onto the website which was built using Django architecture.
9. The predictions given by the model indicates the highest possible price for that particular diamond. 

# Drive Link (Video): 
https://drive.google.com/file/d/1vphSAMUETPbYAIx0qu5DFPdBSGIaB00D/view?usp=sharing

# Conclusion:
The Diamond price predicting model was successfully implemented and was able to accurately predict the price of diamonds with an MAE (mean absolute error) value of 0.8469.
The model was later loaded onto a website using Django architecture to provide a seamless User-interface.

