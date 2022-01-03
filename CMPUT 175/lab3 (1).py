#--------------------------------------------------------
# Lab 3: Check if squares stored in text files are MAGIC.
# 
# Author: CMPUT 175 team
#--------------------------------------------------------

from lab3_MagicSquare import MagicSquare
import os
os.chdir(os.path.dirname(__file__))

def getData(filename):
    '''
    Reads in and returns the data from filename, which contains an unknown
    number of lines of text. Groups the data from each line together.
    
    Input:
       filename (str) - name of text file containing data
    
    Returns:
       data (list of lists) - data read from filename
    '''
    data = []
    fin = open(filename, 'r')
    fileContents = fin.readlines()
    for line in fileContents:
        line = line.strip()  # be sure to remove '\n' along with any other leading and trailing whitespace
        data.append(line.split(','))
    fin.close() 
    
    return data


def populateSquare(values):
    '''
    Creates and populates an instance of MagicSquare with values.
    
    Inputs:
       values (list of lists) - contains data to place in MagicSquare,
                                where data is grouped by row
       
    Returns: MagicSquare
    '''

    square = MagicSquare(len(values))
    for rowIndex in range(len(values)):
        for colIndex in range(len(values)):
            square.update(rowIndex, colIndex, int(values[rowIndex][colIndex])) 
    return square


def displayResult(square, filename):
    '''
    Displays contents of square, along with whether it is magic or not
    
    Input:
       square (MagicSquare) - square filled with data
       filename (str) - name of text file where data was retrieved from
    
    Returns: None
    '''    
    print('\nReading from {}...'.format(filename))
    square.drawSquare()
    if square.isMagic():
        print('This square is MAGIC')
    else:
        print('Nothing magic about this square')
    print('='*25)
    

def main():
    '''
    Reads data from textfiles in current directory and identifies MAGIC squares
    Inputs: n/a
    Returns: None
    '''
    for file in os.listdir():   # iterates through all items in the folder where this Python file is located     
        if file.endswith('.txt'):  # only considers the .txt files
            squareValues = getData(file)
            filledSquare = populateSquare(squareValues)
            displayResult(filledSquare, file)
        

if __name__ == '__main__':
    main()
