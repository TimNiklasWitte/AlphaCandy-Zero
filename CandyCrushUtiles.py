CANDY_BUFF_HEIGHT = 5

def isNormalCandy(candyID):
    return 1 <= candyID <= 6
    
def isWrappedCandyID(candyID):
    return 7 <= candyID <= 12
    
def isHorizontalStrippedCandy(candyID):
    return 13 <= candyID <= 18
 
def isVerticalStrippedCandy(candyID):
    return 19 <= candyID <= 24


def convert_normalCandyID_name(candyID):
    if candyID == 1:
        return "Red"
    elif candyID == 2:
        return "Orange"
    elif candyID == 3:
        return "Yellow"
    elif candyID == 4:
        return "Green"
    elif candyID == 5:
        return "Blue"
    elif candyID == 6:
        return "Purple"
    
def getWrappedCandyID(candyID):
    return candyID + 6

def getHorizontalStrippedCandyID(candyID):
    return candyID + 12
    
def getVerticalStrippedCandyID(candyID):
    return candyID + 18

    
def convertWrappedCandy_toNormal(candyID):
    return candyID - 6
    
def convertHorizontalStrippedCandy_toNormal(candyID):
    return candyID - 12
    
def convertVerticalStrippedCandy_toNormal(candyID):
    return candyID - 18

def equalCandys(candyID_1, candyID_2):
       
    if isNormalCandy(candyID_1):
        id1 = candyID_1
    elif isWrappedCandyID(candyID_1):
        id1 = convertWrappedCandy_toNormal(candyID_1)
    elif isHorizontalStrippedCandy(candyID_1):
        id1 = convertHorizontalStrippedCandy_toNormal(candyID_1)
    elif isVerticalStrippedCandy(candyID_1):
        id1 = convertVerticalStrippedCandy_toNormal(candyID_1)
    else:
        return False

    if isNormalCandy(candyID_2):
        id2 = candyID_2
    elif isWrappedCandyID(candyID_2):
        id2 = convertWrappedCandy_toNormal(candyID_2)
    elif isHorizontalStrippedCandy(candyID_2):
        id2 = convertHorizontalStrippedCandy_toNormal(candyID_2)
    elif isVerticalStrippedCandy(candyID_2):
        id2 = convertVerticalStrippedCandy_toNormal(candyID_2)
    else:
        return False

    return id1 == id2


FIELD_SIZE = 8
NUM_DIRECTIONS = 4

def isValidAction(action):

    fieldID = action // NUM_DIRECTIONS

    direction = action % NUM_DIRECTIONS

    x = fieldID // FIELD_SIZE
    y = (fieldID % FIELD_SIZE) + CANDY_BUFF_HEIGHT

    # Swap candy
    x_swap = x # attention: numpy x->y are swapped
    y_swap = y # attention: numpy x->y are swapped
    # top
    if direction == 0:
        y_swap += -1
    # down
    elif direction == 2: 
        y_swap += 1
    # right 
    elif direction == 1:
        x_swap += 1
    # left 
    elif direction == 3:
        x_swap += -1

    return isValidIndex(x,y) and isValidIndex(x_swap, y_swap)

def isValidIndex(x, y):
    if FIELD_SIZE <= x or FIELD_SIZE + CANDY_BUFF_HEIGHT <= y or x < 0 or y < CANDY_BUFF_HEIGHT:
        return False
        
    return True

COLOR_BOMB_CANDY_ID = 25
NUM_CANDIES = COLOR_BOMB_CANDY_ID