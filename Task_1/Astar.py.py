import cv2 as cv
import numpy as np
import math

# This function determines if the considered point is inside the bounds of the image.
def inRange(img,x,y):
    return x>=0 and x<img.shape[0] and y>=0 and y<img.shape[1]

# Takes in the processed image.
image = cv.imread("image_gray2.png")

image = cv.resize(image, (610, 338), interpolation = cv.INTER_LINEAR)

# We create a copy of the original image to not accidentally change data. This is more of a stretched cautionary step.
temp = image.copy()

# Declaring the color values we will be using throughout this code
blue = [255, 0, 0]
white = [255, 255, 255]
black = [0, 0, 0]
found = [0, 255, 255]

# This stores the startXY and endXy co-ordinates
startXY = (274,354)
endXY = (85, 257)

# open_set = Stores the pixels that have been reached but the neighbors have not been explored yet.
# closed_set = Stores the pixels that have been reached and the neighbors have been explored. 
open_set = set([startXY])
closed_set = set([])

# g is the cost to reach the corresponding vertex from startXY
g = {}

# parents stores the parent of a particular pixel (Implies that it was reached through the parent pixel)
parents = {}

# the cost to reach startXY is 0
g[startXY] = 0 

# declaring the values for the coordinates individually
startX = startXY[0]
startY = startXY[1]
endX = endXY[0]
endY = endXY[1]

# parent of startXY is itself. We will later use this to reconstruct the path
parents[startXY] = startXY

# setting the flag to zero. If set to zero, that implies the optimum path has been found
flag = 0

# we mark the explored parts in blue
temp[(startX, startY)] = blue

# The A* Algorithm's implementation. Starts with a loop that proceeds till the entire image has been explored.
# If this happens then open_set will have zero elements.
while len(open_set) != 0:
    # n is a node that we are about to consider
    n = None
    
    # The loop checks for the minimum cost + heuristic node in the open_set
    for (i, j) in open_set:
        # We ran into a problem of overestimating the heuristic function when we declared the heuristic as
        # |x - x0| + |y - y0|. This was because, the path can be traversed even diagonally. Which might lead to
        # shorter paths. Thus, we use sqrt((x-x0)^2 + (y-y0)^2) as the heuristic.
        if n == None or g[(i, j)] +  math.sqrt((endX-i)**2+ (endY-j)**2) < g[n] + math.sqrt((n[0]-endX)**2 + (n[1]-endY)**2):
            n = (i, j)
    
    # Set the coordinates to n[0] and n[1]
    i = n[0]
    j = n[1]
    
    # These will iterate through the neighbours
    for (r,s) in [(0, 1), (1, 0), (1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1)]:
        
        if inRange(temp,r+i,s+j):
            
            # Checks is (r+i, s+j) has already been explored. And also if it is traversable. That is it is not black
            if ((r+i, s+j) not in open_set) and ((r+i, s+j) not in closed_set) and ((temp[i+r, j+s] != black).any()):
                open_set.add((r+i, s+j))
                temp[i+r, j+s] = blue
                parents[(r+i, s+j)] = (i, j)
                g[(r+i, j+s)] = g[(i, j)] + 1
            
            # Runs this condition is the node has already been traversed once before
            elif (temp[i+r, j+s] != black).any():
                # Checks if the cost is lesser than before
                if g[(r+i, s+j)]> g[(i, j)] + 1:
                    parents[(r+i, s+j)] = (i, j)
                    g[(r+i, j+s)] = g[(i, j)] + 1
                    
                    # Opens up the node since it has a newer path allocated
                    if (r+i, s+j) in closed_set:
                        closed_set.remove((r+i, s+j))
                        open_set.add((r+i, s+j))
            
            # if the endXY is reached, then flag is set to 1            
            if (i+r, j+s) == endXY:
                flag = 1
                break
            '''
            # Uncomment this to show animated display.
            cv.waitKey(1)
            cv.imshow("Image", temp)
            '''
            
    # once done exploring all the neighbors, we add n to the closed set.
    open_set.remove(n)
    closed_set.add(n)
    if flag == 1:
        break

cv.waitKey(1)
count = 0

# The code to reconstruct the path
if flag == 1:
    reconst_path = []
    n = endXY
    while parents[n] != n:
        reconst_path.append(n)
        n = parents[n]

    reconst_path.append(startXY)
    reconst_path.reverse()
    
    for i in reconst_path:
        count += 1
        temp[i[0]][i[1]] = found

# prints the path length
print(count)
cv.imshow("Image", temp)
cv.waitKey(0)
cv.destroyAllWindows()