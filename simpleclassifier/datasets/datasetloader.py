import os
import cv2
import numpy as np


class SimpleDatasetLoader:
    def __init__(self,preprocessors=None):
        #store the image preprocessor

        self.preprocessors=preprocessors

        if self.preprocessors is None:
            self.preprocessors=[]


    def load(self,image_paths,verbose=1):

        #features and labels

        data=[]
        labels=[]

        #load over the input images

        for (i,image_path) in enumerate(image_paths):
            print("[INFO]: Loading {}".format(image_path))
            image=cv2.imread(image_path)
            label=image_path.split(os.path.sep)[-2]

            if self.preprocessors is not None:

                for p in self.preprocessors:
                    image=p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose >0 and i >0 and (i +1) % verbose ==0:
                print("[INFO] processed {}/{}".format(i+1,image_paths))

        
        return (np.array(data),np.array(labels))


