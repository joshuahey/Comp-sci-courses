#Memory game
#Version 2:
# - no matching of tiles
# - initally placeholder image
# - when clicked -> reveal image
# - keep all images open
# - keep score
# - close when all 16 have been opened (without matching pairs)

import pygame,os
from random import *
import time
os.chdir(os.path.dirname(__file__))

def main():
    """
    Initialize Pygame, draw initial objects, and run game loop
    """

    pygame.init()

    size = (560, 450)
    title = "Memory"

    #Create the window
    surface = pygame.display.set_mode(size)
    pygame.display.set_caption(title)

    #Load Images
    imageTitles = ["image1.bmp", "image2.bmp", "image3.bmp", "image4.bmp", "image5.bmp", "image6.bmp", "image7.bmp", "image8.bmp"]
    images = []
    for imageTitle in imageTitles:
        image = pygame.image.load(imageTitle)
        images.append(image)
        images.append(image)

    #Create and initialize objects
    game = memory(surface, images)

    #Draw objects
    game.draw()
    pygame.display.update()

    while(True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP and not game.gameOver:
                game.handleMouseUp(event.pos)

        game.updateScore()
        pygame.display.update()

class memory:
    boardSize = 4
    gap = 10        #Gap between tiles

    def __init__(self, surface, images):
        """Initializes the board
        images -> randomized list of all 16 image objects
        """
        self.gameOver = False
        self.startTime = round(time.time())     #time since epoch rounded to seconds

        self.surface = surface
        #Shuffle the list of images
        shuffle(images)
        self.images = images

        #tileWidth = (surface.get_width() - ((memory.boardSize + 2)*memory.gap))//(memory.boardSize + 1)
        #tileHeight = (surface.get_height() - ((memory.boardSize + 1)*memory.gap))//memory.boardSize
        tileWidth = images[0].get_width()
        tileHeight = images[0].get_height()

        #Create the board
        self.board = []
        for row in range(0, memory.boardSize):
            rows = []
            for column in range(0, memory.boardSize):
                x = ((column + 1) * memory.gap) + (column * tileWidth)
                y = ((row + 1) * memory.gap) + (row * tileHeight)
                tilePosition = [x, y, tileWidth, tileHeight]

                #Select a random image from the images list and delete the image so it can't be used again (ensures only 2 of each image)
                imageIndex = randrange(0, len(self.images))
                image = self.images[imageIndex]
                self.images.remove(image)

                #Create tile
                tile = Tile(tilePosition, image, surface)

                #Add the tile to the row
                rows.append(tile)

            #Add the row list to the board
            self.board.append(rows)

    def draw(self):
        """Draws the tiles"""
        for row in self.board:
            for tile in row:
                tile.draw()
        #Update the screen with the images
        pygame.display.flip()

    def handleMouseUp(self, mousePosition):
        """Called when the mouse is released after a clicked
        mousePosition -> (x, y)
        """
        #Check if the mouse is on any of the tiles
        for row in self.board:
            for tile in row:
                tile.reveal(mousePosition)
        self.checkGameOver()

    def checkGameOver(self):
        """Checks if the all the tiles have been revealed or not"""
        revealedTilesCount = 0
        for row in self.board:
            for tile in row:
                if tile.revealed:
                    revealedTilesCount += 1
        if revealedTilesCount == 16:
            self.gameOver = True

    def updateScore(self):
        """Update the score in the top right corner with the seconds passed since beginning of game"""
        if not self.gameOver:
            #Draw a rect in the same position as the score to clear that area before updating the score
            pygame.draw.rect(self.surface, pygame.Color("Black"), (self.surface.get_width() - 50, 0, 100, 100))

            #Get the current time to calculate the time difference since the start of the game
            currentTime = round(time.time())
            score = currentTime - self.startTime

            myFont = pygame.font.SysFont(None, 60)
            self.surface.blit(myFont.render(str(score), 1, pygame.Color("white")), (self.surface.get_width() - 50, 0))

class Tile:
    #Load the Question Mark placeholder image that will be on every tile at the beginning
    placeHolderImage = pygame.image.load("image0.bmp")

    def __init__(self, tilePosition, image, surface):
        """Initializes the tile"""
        self.x = tilePosition[0]
        self.y = tilePosition[1]
        self.width = tilePosition[2]
        self.height = tilePosition[3]
        self.image = image
        self.surface = surface
        self.revealed = False       #False = placeHolderImage still showing

    def draw(self, image=placeHolderImage):
        """Draws the tile
        image -> the image that the tile has - placeHolderImage if no image passed in
        """
        self.surface.blit(image, (self.x, self.y))

    def reveal(self, mousePosition):
        """Replaces the placeHolderImage of the tile with another assigned image
        mousePosition -> (x,y)
        """
        #Only checks if the mouse was clicked on this tile if it hasn't been revealed yet
        if not self.revealed:

            mouseX = mousePosition[0]
            mouseY = mousePosition[1]

            #Check if the mouse click was on the tile
            if mouseX >= self.x and mouseX <= self.x + self.width:
                if mouseY >= self.y and mouseY <= self.y + self.height:
                    #Update tile with the assigned image instead of placeHolderImage
                    self.draw(self.image)
                    self.revealed = True

main()