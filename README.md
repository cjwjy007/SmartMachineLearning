# SmartMachineLearning
用法： 1.导入csv train_df = pd.read_csv(file_path)

2.初始化smart_ml smart_ml = SmartML(train_df=train_df)

3.调用auto_learn smart_ml.auto_learn()

注：预处理结束后会在根目录生成preprocessed_df.csv，即预处理结束后的dataframe 如下调用可以直接跳过初始化工作，进行训练

train_df = pd.read_csv(file_path)

preprocessed_df = pd.read_csv("preprocessed_df.csv")

smart_ml = SmartML(preprocessed_df=preprocessed_df)

smart_ml.auto_learn()
