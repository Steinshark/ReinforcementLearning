import time 
import random

text = open("SnakeConcurrentIMG.py",'r').read()


while text:
    write_file  = open("test.py",'a')
    time.sleep(.01*random.randint(1,6))
    write_file.write(text[1])
    text = text[1:] 
    write_file.close()
