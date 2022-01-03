# x=input()
# y=input()
# print(x+" and "+ y +" are equal!" if x==y else x+" and "+ y +" are not equal")

# a=[]
# for i in range(0,3):
#     a.append(input())
# a.sort()
# print("The middle element is ",a.sort())

# x=int(input())
# if x<=0:
#     print("The temperature is freezing")
# elif x in range(10):
#     print("The temperature is cold")
# elif x in range(10,20):
#     print("The temperature is chilly")
# elif x in range(20,30):
#     print("The temperature is warm")
# elif x in range(30,40):
#     print("The temperature is hot")
# else:
#     print("The temperature is extremely hot!")


#program to find which quadrant/axis does a pair of coordinates lie

# x=int(input("Input the value for the X coordinate:")) # prompt for x coordinate 
# y=int(input("Input the value for the Y coordinate:")) # prompt for y coordinate 

#conditional statements to show quadrants/axises where the inputted coordinates lie

def coord(x,y):
    if x > 0 and y > 0: 
        print ("The point (" ,x,",",y, ") lies in quadrant I") 
    elif x < 0 and y > 0: 
        print ("The point (" ,x,",",y, ") lies in quadrant II") 
    elif x < 0 and y < 0: 
        print ("The point (" ,x,",",y, ") lies in quadrant III") 
    elif x > 0 and y < 0: 
        print ("The point (" ,x,",",y, ") lies in quadrant IV") 

    elif x == 0 and y > 0 or x == 0 and y < 0 or y == 0 and x < 0 or y == 0 and x > 0 : 
        print ("The point (" ,x,"," ,y, ") lies on a border between two quadrants") 
    else:
        print ("The point (" ,x,",",y, ") at the origin") 
    return ''

x=int(input("Input the value for the X coordinate:")) # prompt for x coordinate 
y=int(input("Input the value for the Y coordinate:")) # prompt for y coordinate 
coord(x,y)

