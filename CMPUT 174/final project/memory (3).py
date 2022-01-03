# Pre-Poke Framework
# Implements a general game template for games with animation
# You must use this template for all your graphical lab assignments
# and you are only allowed to inlclude additional modules that are part of
# the Python Standard Library; no other modules are allowed

import os,pygame
from random import randrange
os.chdir(os.path.dirname(__file__))

# User-defined functions
def main():
   # initialize all pygame modules (some need initialization)
   pygame.init()
   # create a pygame display window
   pygame.display.set_mode((500, 400))
   # set the title of the display window
   pygame.display.set_caption('Memory')   
   # get the display surface
   w_surface = pygame.display.get_surface() 
   # create a game object
   game = Game(w_surface)
   # start the main game loop by calling the play method on the game object
   game.play() 
   # quit pygame and clean up the pygame window
   pygame.quit() 

# User-defined classes
class Game:
   # An object in this class represents a complete game.
   def __init__(self, surface):
      # Initialize a Game.
      # - self is the Game to initialize
      # - surface is the display window surface object
      # === objects that are part of every game that we will discuss
      self.surface = surface
      self.bg_color = pygame.Color('black')    
      self.FPS = 60
      self.game_Clock = pygame.time.Clock()
      self.close_clicked,self.continue_game = False,True
      # === game specific objects
      self.board_size = 4
      self.board = []  
      self.timer=True
      self.max_frames = 150
      self.frame_counter = 0
      self.image_list =['image'+str(i)+'.bmp' for i in range(1,9)]
      self.images=[]
      self.revealed_images=[]
      #loop for making image list
      for image in self.image_list:
         j=pygame.image.load(image)
         self.images.append(j)
         self.images.append(j)
      self.create_board()
      
   def create_board(self):
      """[creating board]
      width= width of the board
      height= width of the board
      """
      width = self.surface.get_width()//(self.board_size+1)
      height = self.surface.get_height()//self.board_size
      for row_index in range(0,self.board_size): 
         row = []
         for col_index in range(0,self.board_size):
            x = col_index * (width)+1
            y = row_index * (height)+1
            image=self.images[randrange(0,len(self.images))] #random image selected
            self.images.remove(image) #image selected removed after selection
            tile = Tile(x,y,width,height,image,self.surface) #tile object created
            row.append(tile)
         self.board.append(row)

   def play(self):
      # Play the game until the player presses the close box.
      # - self is the Game that should be continued or not.
      while not self.close_clicked:  # until player clicks close box
         # play frame
         self.handle_events()
         self.draw()     
         self.time_elasped_hide()    
         if self.continue_game:
            self.update()
            self.decide_continue()
         self.game_Clock.tick(self.FPS) # run at most with FPS Frames Per Second 

   def handle_events(self):
      # Handle each user event by changing the game state appropriately.
      # - self is the Game whose events will be handled
      events = pygame.event.get()
      for event in events:
         if event.type == pygame.QUIT:
            self.close_clicked = True
         if event.type == pygame.MOUSEBUTTONUP and self.continue_game:
            self.handle_mouse_up(event.pos)

   def draw(self):
      # Draw all game objects.
      # - self is the Game to draw  
      self.surface.fill(self.bg_color) # clear the display surface first
      for row in self.board:
         for tile in row:
            tile.draw()
      if self.timer:
         self.score = pygame.time.get_ticks()//1000
         font=pygame.font.Font(None,74)
         text=font.render(str(self.score),1,"white")
         self.surface.blit(text,(440,0))
         pygame.display.update() # make the updated surface appear on the display

   def handle_mouse_up(self,position):
      """[mouse up method]
      revealed_images=list appending two tiles
      Args:
          position ([tuple]): [mouse position]
      """
      if len(self.revealed_images) < 2:
         for row in self.board:
            for tile in row:
               if tile.select(position):
                  self.revealed_images.append(tile)
                  if len(self.revealed_images) == 2:
                     self.check_matching() 

   def check_matching(self):
      """[checking if tiles matches in revealed_images list]
      """
      if self.revealed_images[0].same_tiles(self.revealed_images[1]):
         [self.revealed_images[i].reveal() for i in (0,1)]
      else:
         [self.revealed_images[i].hide() for i in (0,1)]
      self.revealed_images.clear()
         
   def time_elasped_hide(self):
      """[check]
      """
      for row in self.board:
         for tile in row:
            tile.time_check()
   def update(self):
      # Update the game objects for the next frame.
      # - self is the Game to update
      pass
   def decide_continue(self):
      # Check and remember if the game should continue
      # - self is the Game to check
      tilesrevealed = 0
      for row in self.board:
         for tile in row:
            if tile.expose:
               tilesrevealed += 1
      if tilesrevealed == 16:
         self.continue_game= False
         self.timer=False

class Tile:
   def __init__(self,x,y,width,height,image,surface):
      self.rect = pygame.Rect(x,y,width,height)
      self.x,self.y,self.width,self.height=x,y,width,height
      self.surface = surface
      self.image=image
      self.expose=False
      self.time_elasped=0

   def draw(self):
      """[draw tiles]
      """
      color = pygame.Color('black')
      border_width =5
      pygame.draw.rect(self.surface,color,self.rect,border_width)
      if self.expose:
         self.surface.blit(self.image, (self.x,self.y))
      else:
         self.surface.blit(pygame.image.load("image0.bmp"), (self.x,self.y)) 

   def reveal(self):
      """[expose tile]
      """
      self.expose=True
   def hide(self):
      """[start timer to hide]
      """
      self.time_elasped=pygame.time.get_ticks()//1000

   def time_check(self):
      """[keeps a limit for the timer to hide]
      we start a timer in the main class time_elasped_hide, and once we click two tiles the timer for the tiles start in hide()
      in this function we check when the timer time_elasped is atmost the intial timer- (any value)
      """
      if self.time_elasped!=None and self.time_elasped < (pygame.time.get_ticks()//1000)-0.2:
         self.expose = False
         self.time_elasped = None

   def select(self,position):
      """[check if valid click]
      Args:
          position ([tuple]): [mouse pos]
      Returns:
          [bool]: [checks valid click]
      """
      valid_click = False
      if self.rect.collidepoint(position):   
         if not self.expose:
            valid_click = True
            self.reveal()
         else:
            valid_click = False
      return valid_click     
                     
   def same_tiles(self, other): 
      """[check if two tiles are equal]
      Args:
          other ([Tile]): [tile]
      Returns:
          [bool]: [check condition]
      """
      return self.image == other.image
main()