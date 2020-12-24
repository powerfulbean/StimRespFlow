import librosa  
import librosa.display
import StellarDataIO as IO


wav_path = ['D:\\appendix\\A中文语音识别\\data_thchs30\\data\\']
label_file = 'D:\\appendix\\A中文语音识别\\data_thchs30\\doc\\trans\\train.word.txt'
test_dile_path = ['D:\\appendix\\EEGDataSetV1.1\\wave\\InterfereWave\\',
                  'D:\\appendix\\EEGDataSetV1.1\\wave\\NormalWave\\',
                  'D:\\appendix\\EEGDataSetV1.1\\wave\\PatientWave\\']
suffix = '.wav'
#data = IO.DataSet()
##data.buildFilelist(data.DataList,wav_path,suffix)
#data.buildFilelist(data.DataList,test_dile_path,'.txt')
#filelist=data.DataList.filelist
#data.buildDataSet([0,1,2])
#dataSet=data.DataSet.trainSet_tag

data2=IO.thchsDataSet()
data2.buildFilelist(data2.DataList,wav_path,suffix)
filelist2=data2.DataList.filelist