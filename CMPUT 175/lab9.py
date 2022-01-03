#******************************************************
# Lab 9: Sorting and Searching
# Author: 
# Collaborators/References:
#******************************************************

##############
##EXERCISE 1
##############
import os
os.chdir(os.path.dirname(__file__))

class Student():
    def __init__(self, idNum, name, mark):
        '''
        Initializes the object properties
        Input:
          idNum (str) - the student's unique university ID, made up of 6 digits
          name (str) - the student's full name
          mark (int) - the student's overall mark in the course
        '''
        assert isinstance(idNum, str) and len(idNum) == 6, 'Invalid student ID'
        assert isinstance(name, str), 'Invalid student name'
        assert isinstance(mark, int), 'Invalid course mark'
        
        self.__id = idNum
        self.__name = name
        self.__mark = mark
	
	
    def getMark(self):
        '''Returns the value of the Student's mark'''
        return self.__mark    


    def __str__(self):
        '''Informal string representation of Student'''
        return ' - {}, {}, {}'.format(self.__id, self.__name, self.__mark)

    def __lt__(self, anotherStudent):
        ''' 
        Checks if the mark of the student is LESS THAN the mark of another 
        Student object
        
        Input: anotherStudent (Student)
        Returns: boolean
	'''
        return self.__mark < anotherStudent.__mark
    

def smallestIndex(data, first, last):
    '''
    Recursively finds the index location of the smallest object in data,
    between the indices first and last.
    
    Inputs:
      data (list) - list of objects
      first (int) - index of beginning of portion list to be considered
      last (int) - index of end of portion of list to be considered
    '''
    if first == last:
        return first
    if data[first] > data[last]:
        first+=1
    else:
        last-=1
    return smallestIndex(data, first, last)
    

def recursiveSelectionSort(data, length, first=0): 
    '''
    Use Selection Sort to RECURSIVELY arrange the objects 
    in a list (data) in descending order.
    
    Inputs:
       data (list) - list of objects to be sorted
       length (int) - number of elements in the portion of data being considered
       first (int) - index of starting element (default value is 0)
    Returns: None
    '''
    # HINTS:
    # What is the base case?
    # Find the minimum index 
    # Swap the data    
    # Recursively call selection sort function 
    if length==1:
        return data
    else:
        min1=smallestIndex(data[first:],first,length-1)
        temp=data[length-1]
        data[length-1]=data[min1]
        data[min1]=temp
        return recursiveSelectionSort(data, length-1)

    
def readStudents():
    # Read the data
    studentFile = open('studentList.txt', 'r')
    
    # Create a Student object corresponding to each line in input file
    students = []
    for student in studentFile:
        fields = student.split(', ')
        ID = fields[0]
        name = fields[1]
        mark = fields[2]
        students.append(Student(ID, name, int(mark)))
        
    # Don't forget to close the file when done with it!
    studentFile.close()
    

    return students


def testStudentSort(students):
    # Print the original data
    print('Original data:')
    for student in students:
        print(student)

    # Sort the students: notice that the selection sort does not create a new list
    recursiveSelectionSort(students, len(students))

    # Print the sorted data
    print('\nSorted data:')
    for student in students:
        print(student)    



##############
##EXERCISE 2
##############
def binarySearch1(key,data,lowerBound,upperBound):
    ''' 
    Finds and returns the position of key in data,
    or returns -1 if key is not in the list
    
    Inputs:
      - key is the target object that we are looking for
      - data is a list of objects, sorted in DECREASING order
      - lowerBound is the lowest index of data
      - upperBound is the highest index of data
    '''
    found = False
    while (not found and lowerBound<=upperBound):
        guessIndex = (upperBound+lowerBound)//2
        if (key == data[guessIndex]):
            found = True
        else:
            if (key < data[guessIndex]):
                lowerBound = guessIndex + 1
            else:
                upperBound = guessIndex - 1
    if (not found):
        guessIndex = -1

    return guessIndex


def binarySearch2(key, data, lowerBound, upperBound):
    ''' 
    RECURSIVELY finds and returns the position of key in data,
    or returns -1 if key is not in the list
    
    Inputs:
      - key is the target object that we are looking for
      - data is a list of objects, sorted in DECREASING order
      - lowerBound is the lowest index of portion of data to be considered
      - upperBound is the highest index of portion of data to be considered
    '''  
  
    if lowerBound==upperBound:
        if key == data[lowerBound]:
            return lowerBound
        else:
            return -1
    else:
        guessIndex = (upperBound+lowerBound)//2
        if key == data[guessIndex]:
            return guessIndex
        elif key > data[guessIndex]:
            return binarySearch2(key, data, lowerBound, guessIndex-1)
        elif key < data[guessIndex]:
            return binarySearch2(key, data, guessIndex+1, upperBound)

def testBinarySearch():
    someList = [9,7,5,3,1,-2,-8]
    print(binarySearch2(9,someList,0,len(someList)-1))
    print(binarySearch2(-8,someList,0,len(someList)-1))
    print(binarySearch2(4,someList,0,len(someList)-1))
  

if  __name__== "__main__":
    #test Exercise 1:
    students = readStudents()
    testStudentSort(students)
    
    #test Exercise 2:
    testBinarySearch()
