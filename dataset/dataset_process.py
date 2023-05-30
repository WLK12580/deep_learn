import pandas as pd
import numpy as np
import csv
from torch.utils.data import Dataset,DataLoader
import skimage.io,skimage.color

class CSVDataset(Dataset):
    def __init__(self,train_file,class_file,transform=None):
        """_summary_
        Args:
            train_file (cvs): 训练文件，格式为csv
            class_file (csv): 类别文件格式为csv
            transform (_type_, optional): _description_. Defaults to None.
        """
        self.train_file=train_file
        self.class_file=class_file
        self.transform=transform
    def read_classes_csv(self):
        try: 
            self.class_data=pd.read_csv(self.class_file)
            self.label={}
            self.classes_key={}
            for key,value in zip(self.class_data["classes"],self.class_data["table"]):
                self.label[value]=key
                self.classes_key[key]=value
        except ValueError as ec:
            raise(ValueError("invalid csv format {} and {}".format(self.class_file,ec)))
        print("self.label={}".format(self.label))
        return self.label
    
    def read_train_csv(self):
        read_result={}
        classes_list=[]
        class_=pd.read_csv(self.class_file)
        classes_types=class_["classes"]
        for i in range(len(classes_types)):
            classes_list.append(classes_types[i])
        print("classes_types=\n{} {}".format(type(classes_list),classes_list))
        try:
            self.train_data=pd.read_csv(self.train_file)
            # print("tarin_data={}".format(self.train_data))
            for index,data in self.train_data.iterrows():
                if data["x2"]<data["x1"]:
                    raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(index, data["x2"], data["x1"]))
                if data["y2"]<data["y1"]:
                    raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(index, data["y2"], data["y1"]))
                if data["label"] not in classes_list: #classes_list:必须为列表
                    raise ValueError("line {} : is not a lable in classes {}".format(index,data["label"]))
                read_result[data["image"]]=({"x1":data["x1"],"y1":data["y1"],"x2":data["x2"],"y2":data["y2"],"label":data["label"]})
            # print("read_result={}".format(read_result)) 
            self.image_name=list(read_result.keys())  #列表
            print("image_name={}".format(self.image_name))  
            return read_result
        except ValueError as ec:
            raise(ValueError("invalid csv format {} and {}".format(self.class_file,ec)))
    def classes_name_to_label(self,name):
        return self.classes_key[name]
        
        
    def __len__(self):
        print("len_dataset={}".format(len(self.image_name)))
        return len(self.image_name)
    
    def load_annotation(self,image_index):#为__getitem__返回数据做前提准备
        annotation_list=self.read_train_csv[self.image_name[image_index]]
        annotations=np.zeros((0,5))
        if len(annotation_list)==0:
            return annotations
        for index,data in enumerate(annotation_list):
            x1=data["x1"]
            y1=data["y1"]
            x2=data["x2"]
            y2=data["y2"]
            if (x2-x1<1) or (y2-y1)<1:
                continue
            anno=np.zeros((1,5))
            anno[0,0]=x1
            anno[0,1]=y1
            anno[0,2]=x2
            anno[0,3]=y2
            anno[0,4]=self.classes_name_to_label[data["label"]]
            
            annotations=np.append(annotations,anno,axis=0)
        
        return annotations
      
    def load_images(self,image_index):   #根据索引获取图像
        img_data=skimage.io.imread(self.image_name[image_index])
        if len(img_data.shape)==2:
            img_data=skimage.color.gray2rgb(img_data)
        return img_data.astype(np.float32)/255
        
    def __getitem__(self,index):
        img=self.load_images(index)
        annot=self.load_annotation(index)
        sample={"img":img,"annot":annot}
        if self.transform:
            sample=self.transform(sample)
        return sample
                   
      
        
        
        
          
    

if __name__=="__main__":
    csvdata=CSVDataset("./dataset_csv/train.csv","./dataset_csv/classes.csv")
    print(csvdata.read_classes_csv())
    csvdata.read_train_csv()