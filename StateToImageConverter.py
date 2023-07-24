import numpy as np
import sys
from PIL import Image

from CandyCrushUtiles import *

#from matplotlib import pyplot as plt

class StateToImageConverter:
    def __init__(self, field_size, candy_buff_height, image_size):
        self.field_size = field_size
        self.image_size = image_size
        self.root_img_path = sys.path[0]+"/Images"

        self.candy_buff_height = candy_buff_height
        
        self.candy_imgs = np.zeros(shape=(NUM_CANDIES + 1, image_size, image_size, 3), dtype=np.float32)

        for candyID in range(1, NUM_CANDIES + 1):
            
            candyID_tmp = candyID
            if isNormalCandy(candyID):
                file_name = convert_normalCandyID_name(candyID)
                img = Image.open(f"{self.root_img_path}/Normal/{file_name}.png")
                    
            elif isWrappedCandyID(candyID):
                candyID_tmp = convertWrappedCandy_toNormal(candyID)
                file_name = convert_normalCandyID_name(candyID_tmp)
                img = Image.open(f"{self.root_img_path}/Wrapped/{file_name}.png") 
                        
            elif isHorizontalStrippedCandy(candyID):
                candyID_tmp = convertHorizontalStrippedCandy_toNormal(candyID)
                file_name = convert_normalCandyID_name(candyID_tmp)
                img = Image.open(f"{self.root_img_path}/Striped/Horizontal/{file_name}.png")

            elif isVerticalStrippedCandy(candyID):
                candyID_tmp = convertVerticalStrippedCandy_toNormal(candyID)
                file_name = convert_normalCandyID_name(candyID_tmp)
                img = Image.open(f"{self.root_img_path}/Striped/Vertical/{file_name}.png")

            elif candyID == COLOR_BOMB_CANDY_ID:
                img = Image.open(f"{self.root_img_path}/ColourBomb/ColourBomb.png")

            size = (self.image_size, self.image_size)

            img = img.convert('RGB')

            img = img.resize(size)
            
            
            #img = np.array(img, dtype=np.float32)
            
            img = np.array(img, dtype=np.float32)
            img = (img / 128.) - 1
            # plt.imshow(img)
            # print(img)
            # plt.show()
            self.candy_imgs[candyID, :, :, :] = img 

        # plt.imshow(self.candy_imgs[1, :, :, :])
        # print(self.candy_imgs[1, :, :, :])
        # plt.show()


    def __call__(self, state):
        
        game_field_img = np.zeros(shape=((self.field_size + self.candy_buff_height)* self.image_size, self.field_size * self.image_size, 3), dtype=np.float32)
        game_field_img[:, :, :] = 0
        for y in range(self.field_size):
            for x in range(self.field_size + self.candy_buff_height):

                candyID = state[x,y]
                
                try:
                    img = self.candy_imgs[candyID, :, :, :]
                    
                    

                    game_field_img[x*self.image_size:(x*self.image_size) + self.image_size, y*self.image_size:(y*self.image_size) + self.image_size, :] = img
                except:
                    pass
        
        return game_field_img