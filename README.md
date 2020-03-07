# StellarBrainwav
A library to boost the implementation of wave data science application
### Libraries:
Numpy, Keras, Struct, etc.

## Some Initial Idea about the design of this framework

For a normal evaluation of an offline model, always three processes are included: Data Preprocessing (cleaning; filtering; resample); Data Epochs Preparation  and Changed to Dataset; Finally run the model;

So maybe at least three "classes" are needed. Which are CRawDataPre; CDataSetPre; CModel;

This framework should help people easily manage these three processes. 
